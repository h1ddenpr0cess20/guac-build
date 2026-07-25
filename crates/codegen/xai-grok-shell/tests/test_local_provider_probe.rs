//! End-to-end context-window detection against stub LM Studio / Ollama servers.
//!
//! The unit tests in `agent::local_provider` cover response parsing; these
//! cover the part that only a socket can: URL derivation from the configured
//! OpenAI-compatible base URL, the real HTTP round trip, and the failure modes
//! a probe must absorb silently (server down, error status, junk body).
//!
//! Payloads are the shapes each project documents for its native API — LM
//! Studio's `GET /api/v0/models`, Ollama's `GET /api/ps` and `POST /api/show`.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use xai_grok_shell::agent::local_provider::{
    LocalProviderKind, clear_probe_cache, probe_context_windows, probe_model_context_window,
};

/// What a stub route returns: an HTTP status and a body.
#[derive(Clone)]
struct Route {
    status: u16,
    body: String,
}

impl Route {
    fn ok(body: serde_json::Value) -> Self {
        Self {
            status: 200,
            body: body.to_string(),
        }
    }

    fn raw(status: u16, body: &str) -> Self {
        Self {
            status,
            body: body.to_owned(),
        }
    }
}

/// A minimal blocking HTTP/1.1 server that answers a fixed routing table.
struct StubServer {
    addr: SocketAddr,
    running: Arc<AtomicBool>,
}

impl StubServer {
    fn start(routes: HashMap<String, Route>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind stub server");
        let addr = listener.local_addr().expect("stub server addr");
        let running = Arc::new(AtomicBool::new(true));
        let stop = Arc::clone(&running);
        thread::spawn(move || {
            for stream in listener.incoming() {
                if !stop.load(Ordering::Relaxed) {
                    return;
                }
                match stream {
                    Ok(stream) => handle(stream, &routes),
                    Err(_) => return,
                }
            }
        });
        Self { addr, running }
    }

    /// The OpenAI-compatible base URL a user would configure, which is what
    /// the probe is given — it derives the native endpoints from this.
    fn base_url(&self) -> String {
        format!("http://{}/v1", self.addr)
    }
}

impl Drop for StubServer {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        // Unblock the accept loop so the thread observes the flag and exits.
        let _ = TcpStream::connect(self.addr);
    }
}

fn handle(stream: TcpStream, routes: &HashMap<String, Route>) {
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    let path = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .to_owned();

    // Drain headers, noting the body length so the request is fully consumed
    // before we reply — otherwise the client can see a broken pipe.
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                if line == "\r\n" || line == "\n" {
                    break;
                }
                if let Some(value) = line
                    .to_ascii_lowercase()
                    .strip_prefix("content-length:")
                    .map(str::trim)
                    .and_then(|v| v.parse::<usize>().ok())
                {
                    content_length = value;
                }
            }
            Err(_) => return,
        }
    }
    if content_length > 0 {
        let mut body = vec![0u8; content_length];
        let _ = reader.read_exact(&mut body);
    }

    let route = routes
        .get(&path)
        .cloned()
        .unwrap_or_else(|| Route::raw(404, "{}"));
    let response = format!(
        "HTTP/1.1 {} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        route.status,
        route.body.len(),
        route.body,
    );
    let mut stream = reader.into_inner();
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

fn routes(pairs: impl IntoIterator<Item = (&'static str, Route)>) -> HashMap<String, Route> {
    pairs.into_iter().map(|(p, r)| (p.to_owned(), r)).collect()
}

/// LM Studio reports both lengths; the one the model was loaded at is the
/// limit the server will actually enforce, so that is what we must detect.
#[test]
fn detects_lm_studio_loaded_context_length() {
    clear_probe_cache();
    let server = StubServer::start(routes([(
        "/api/v0/models",
        Route::ok(serde_json::json!({
            "data": [{
                "id": "google/gemma-4-2b",
                "object": "model",
                "type": "llm",
                "publisher": "lmstudio-community",
                "arch": "gemma3",
                "compatibility_type": "gguf",
                "quantization": "Q4_K_M",
                "state": "loaded",
                "max_context_length": 131072,
                "loaded_context_length": 8192
            }]
        })),
    )]));

    let window = probe_model_context_window(
        LocalProviderKind::LmStudio,
        &server.base_url(),
        "google/gemma-4-2b",
    );
    assert_eq!(
        window,
        Some(8192),
        "should report the loaded length, not the max"
    );
}

/// A model present in LM Studio but not loaded has no enforced length yet, so
/// its maximum is the best available answer.
#[test]
fn falls_back_to_lm_studio_max_when_not_loaded() {
    clear_probe_cache();
    let server = StubServer::start(routes([(
        "/api/v0/models",
        Route::ok(serde_json::json!({
            "data": [{
                "id": "google/gemma-4-2b",
                "state": "not-loaded",
                "max_context_length": 32768
            }]
        })),
    )]));

    assert_eq!(
        probe_model_context_window(
            LocalProviderKind::LmStudio,
            &server.base_url(),
            "google/gemma-4-2b"
        ),
        Some(32768)
    );
}

/// Ollama's `/api/ps` reports the window the running instance was loaded with
/// (`num_ctx`), which is routinely far below the model's maximum.
#[test]
fn detects_ollama_running_context_length() {
    clear_probe_cache();
    let server = StubServer::start(routes([(
        "/api/ps",
        Route::ok(serde_json::json!({
            "models": [{
                "name": "gemma4:2b",
                "model": "gemma4:2b",
                "size": 3_338_801_152_u64,
                "digest": "abc123",
                "context_length": 4096
            }]
        })),
    )]));

    assert_eq!(
        probe_model_context_window(LocalProviderKind::Ollama, &server.base_url(), "gemma4:2b"),
        Some(4096)
    );
}

/// Nothing loaded: fall through to `/api/show`, whose architecture-scoped key
/// carries the model's trained maximum.
#[test]
fn falls_back_to_ollama_show_when_nothing_loaded() {
    clear_probe_cache();
    let server = StubServer::start(routes([
        ("/api/ps", Route::ok(serde_json::json!({ "models": [] }))),
        (
            "/api/show",
            Route::ok(serde_json::json!({
                "model_info": {
                    "general.architecture": "gemma3",
                    "gemma3.context_length": 131072,
                    "gemma3.embedding_length": 2304
                }
            })),
        ),
    ]));

    assert_eq!(
        probe_model_context_window(LocalProviderKind::Ollama, &server.base_url(), "gemma4:2b"),
        Some(131072)
    );
}

/// Config commonly omits the `:latest` tag that Ollama reports.
#[test]
fn matches_ollama_model_without_latest_tag() {
    clear_probe_cache();
    let server = StubServer::start(routes([(
        "/api/ps",
        Route::ok(serde_json::json!({
            "models": [{ "model": "gemma4:latest", "context_length": 16384 }]
        })),
    )]));

    assert_eq!(
        probe_model_context_window(LocalProviderKind::Ollama, &server.base_url(), "gemma4"),
        Some(16384)
    );
}

/// A server that is not running must resolve to "no information" — never an
/// error, and never a hang. This is the common case: the probe runs whether or
/// not the user has the server up.
#[test]
fn absorbs_a_server_that_is_not_running() {
    clear_probe_cache();
    // Bind and immediately drop, so the port is almost certainly closed.
    let addr = {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        listener.local_addr().expect("addr")
    };
    let base_url = format!("http://{addr}/v1");

    assert!(probe_context_windows(LocalProviderKind::LmStudio, &base_url).is_empty());
    assert_eq!(
        probe_model_context_window(LocalProviderKind::Ollama, &base_url, "gemma4:2b"),
        None
    );
}

/// An older build without the native endpoint answers 404; a malformed body is
/// equally possible. Both must degrade to "no information".
#[test]
fn absorbs_error_status_and_junk_bodies() {
    clear_probe_cache();
    let missing = StubServer::start(routes([("/unrelated", Route::ok(serde_json::json!({})))]));
    assert!(probe_context_windows(LocalProviderKind::LmStudio, &missing.base_url()).is_empty());

    clear_probe_cache();
    let junk = StubServer::start(routes([(
        "/api/v0/models",
        Route::raw(200, "this is not json"),
    )]));
    assert!(probe_context_windows(LocalProviderKind::LmStudio, &junk.base_url()).is_empty());

    clear_probe_cache();
    let boom = StubServer::start(routes([(
        "/api/ps",
        Route::raw(500, "{\"error\":\"internal\"}"),
    )]));
    assert!(probe_context_windows(LocalProviderKind::Ollama, &boom.base_url()).is_empty());
}

/// The probe is handed the OpenAI-compatible base URL, so it must strip that
/// suffix to reach the native API mounted at the server root.
#[test]
fn derives_native_endpoint_from_openai_base_url() {
    clear_probe_cache();
    let server = StubServer::start(routes([(
        "/api/v0/models",
        Route::ok(serde_json::json!({
            "data": [{ "id": "m", "state": "loaded", "loaded_context_length": 2048 }]
        })),
    )]));

    // With the /v1 suffix, without it, and with a trailing slash.
    let root = server.base_url().trim_end_matches("/v1").to_owned();
    for base in [server.base_url(), root.clone(), format!("{root}/v1/")] {
        clear_probe_cache();
        let windows = probe_context_windows(LocalProviderKind::LmStudio, &base);
        assert_eq!(windows.get("m"), Some(&2048), "failed for base_url {base}");
    }
}
