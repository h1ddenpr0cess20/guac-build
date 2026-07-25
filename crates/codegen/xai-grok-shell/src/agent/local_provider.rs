//! Built-in support for locally served models — LM Studio and Ollama.
//!
//! Both expose an OpenAI-compatible surface (including `/v1/responses`, which
//! is the backend this client speaks), so the only things a user should have to
//! supply are which server and which model. This module provides the rest:
//!
//! * **Presets** — base URL, API backend, and the placeholder bearer both
//!   servers accept, so `model_provider = "lmstudio"` works with no
//!   `[model_providers.*]` block. See [`preset`].
//! * **Context-window detection** — neither server reports a context length on
//!   its OpenAI-compatible `/v1/models`, so a window left undetected would fall
//!   back to a hosted-scale default (256K) and overrun a window the user
//!   actually set to 8K. Each server's native API does report it, and
//!   [`probe_context_windows`] reads it from there.
//! * **Small-window budgets** — see [`apply_local_provider_defaults`], which
//!   feeds the detected window into the compaction and tool-output policy in
//!   `xai_token_estimation`.
//!
//! Probes are best-effort: a server that is not running, an old build without
//! the native endpoint, or a malformed body all resolve to "no information",
//! and the caller keeps whatever it already had.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use indexmap::IndexMap;

use super::config::{Config, EnvKeys, ModelEntry};
use super::model_providers::ModelProviderConfig;
use crate::sampling::ApiBackend;

/// How long a probe result is reused before the server is asked again. Long
/// enough that repeated config loads during startup cost one request, short
/// enough that reloading a model at a different context length in LM Studio is
/// picked up within the same session.
const PROBE_TTL: Duration = Duration::from_secs(60);

/// Per-request timeout for probes. These are loopback calls against a server
/// that is either up or refusing connections, so the budget only has to cover
/// a server that is up but busy loading a model.
const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// A locally served OpenAI-compatible model server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LocalProviderKind {
    LmStudio,
    Ollama,
}

impl LocalProviderKind {
    /// The canonical `[model_providers.<id>]` name.
    pub fn id(self) -> &'static str {
        match self {
            Self::LmStudio => "lmstudio",
            Self::Ollama => "ollama",
        }
    }

    /// Human-readable name for logs and warnings.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::LmStudio => "LM Studio",
            Self::Ollama => "Ollama",
        }
    }

    /// The server's default OpenAI-compatible base URL.
    pub fn default_base_url(self) -> &'static str {
        match self {
            Self::LmStudio => "http://localhost:1234/v1",
            Self::Ollama => "http://localhost:11434/v1",
        }
    }

    /// Default listen port, used to recognize a hand-written base URL.
    fn default_port(self) -> u16 {
        match self {
            Self::LmStudio => 1234,
            Self::Ollama => 11434,
        }
    }

    /// Environment variables that may carry a real key. Neither server
    /// authenticates by default, but both are commonly fronted by a proxy that
    /// does, and LM Studio can require one explicitly.
    pub fn env_keys(self) -> EnvKeys {
        match self {
            Self::LmStudio => EnvKeys::new(["LMSTUDIO_API_KEY", "LM_STUDIO_API_KEY"]),
            Self::Ollama => EnvKeys::new(["OLLAMA_API_KEY"]),
        }
    }

    /// The bearer to send when no real key is configured.
    ///
    /// Both servers ignore the value but the client must send something: a
    /// model on a non-first-party base URL with no credential at all is failed
    /// closed by `resolve_model_list`, which would make every local model
    /// unusable. These are the placeholders each project's own documentation
    /// uses.
    pub fn placeholder_api_key(self) -> &'static str {
        match self {
            Self::LmStudio => "lm-studio",
            Self::Ollama => "ollama",
        }
    }

    /// Recognize a provider id written in config, tolerating the common
    /// spellings of the LM Studio one.
    pub fn from_provider_id(id: &str) -> Option<Self> {
        match id.trim().to_ascii_lowercase().replace(['-', '_', ' '], "") {
            ref s if s == "lmstudio" => Some(Self::LmStudio),
            ref s if s == "ollama" => Some(Self::Ollama),
            _ => None,
        }
    }

    /// Recognize a base URL as one of these servers, by host token or by the
    /// project's default port. Used for models configured with a bare
    /// `base_url` and no `model_provider`, and for `GROK_MODELS_BASE_URL`.
    pub fn from_base_url(base_url: &str) -> Option<Self> {
        let lowered = base_url.trim().to_ascii_lowercase();
        if lowered.contains("lmstudio") || lowered.contains("lm-studio") {
            return Some(Self::LmStudio);
        }
        if lowered.contains("ollama") {
            return Some(Self::Ollama);
        }
        let is_loopback = [
            "localhost",
            "127.0.0.1",
            "[::1]",
            "0.0.0.0",
            "host.docker.internal",
        ]
        .iter()
        .any(|h| lowered.contains(h));
        if !is_loopback {
            return None;
        }
        [Self::LmStudio, Self::Ollama]
            .into_iter()
            .find(|kind| lowered.contains(&format!(":{}", kind.default_port())))
    }
}

/// The provider defaults injected when a model names one of these providers.
///
/// `context_window` is deliberately left unset: the real value comes from
/// [`probe_context_windows`] per model, and a provider-wide guess would
/// silently win over detection for every model that shares the provider.
pub fn preset(kind: LocalProviderKind) -> ModelProviderConfig {
    let env_keys = kind.env_keys();
    // Only fall back to the placeholder when no real key is present: a static
    // `api_key` outranks `env_key` at resolve time, so setting both
    // unconditionally would make the env var unreachable.
    let api_key = env_keys
        .resolve_value()
        .is_none()
        .then(|| kind.placeholder_api_key().to_owned());
    ModelProviderConfig {
        base_url: Some(kind.default_base_url().to_owned()),
        api_base_url: None,
        env_key: Some(env_keys),
        api_key,
        api_backend: Some(ApiBackend::Responses),
        extra_headers: IndexMap::new(),
        query_params: IndexMap::new(),
        env_http_headers: IndexMap::new(),
        auth_provider: None,
        auth: None,
        context_window: None,
    }
}

/// Add presets for every local provider a model refers to but no
/// `[model_providers.*]` block defines.
///
/// User blocks are never replaced. A partial block — `[model_providers.ollama]`
/// with only `base_url`, say — has its unset fields filled from the preset, so
/// pointing at a non-default port does not also cost the backend and
/// credential defaults.
pub fn inject_presets(
    providers: &mut IndexMap<String, ModelProviderConfig>,
    referenced_ids: impl IntoIterator<Item = String>,
) {
    for id in referenced_ids {
        let Some(kind) = LocalProviderKind::from_provider_id(&id) else {
            continue;
        };
        match providers.get_mut(&id) {
            Some(existing) => fill_unset_from_preset(existing, kind),
            None => {
                providers.insert(id, preset(kind));
            }
        }
    }
}

/// Fill a user-declared provider's unset fields from the preset.
fn fill_unset_from_preset(provider: &mut ModelProviderConfig, kind: LocalProviderKind) {
    let preset = preset(kind);
    if provider.base_url.is_none() {
        provider.base_url = preset.base_url;
    }
    if provider.api_backend.is_none() {
        provider.api_backend = preset.api_backend;
    }
    // Credentials are filled only when the user supplied none at all, so a
    // preset placeholder can never shadow a real key or auth helper.
    let has_credential = provider
        .api_key
        .as_deref()
        .is_some_and(|k| !k.trim().is_empty())
        || provider
            .env_key
            .as_ref()
            .and_then(EnvKeys::primary)
            .is_some()
        || provider.auth_provider.is_some()
        || provider.auth.is_some();
    if !has_credential {
        provider.env_key = preset.env_key;
        provider.api_key = preset.api_key;
    }
}

/// Context windows keyed by the model id the server reports.
pub type DetectedWindows = HashMap<String, u64>;

/// Strip the OpenAI-compatible suffix to get the server root, where both
/// projects mount their native APIs. `http://localhost:1234/v1` →
/// `http://localhost:1234`.
fn server_root(base_url: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    trimmed.strip_suffix("/v1").unwrap_or(trimmed).to_owned()
}

/// Normalize a model id for comparison. Ollama reports `llama3:latest` where
/// config often says `llama3`, and LM Studio ids are case-insensitive paths.
fn normalize_model_id(model: &str) -> String {
    let lowered = model.trim().to_ascii_lowercase();
    lowered
        .strip_suffix(":latest")
        .unwrap_or(&lowered)
        .to_owned()
}

/// Look a model up in a probe result, exactly first, then normalized.
pub fn lookup_window(windows: &DetectedWindows, model: &str) -> Option<u64> {
    if let Some(window) = windows.get(model) {
        return Some(*window);
    }
    let target = normalize_model_id(model);
    windows
        .iter()
        .find(|(id, _)| normalize_model_id(id) == target)
        .map(|(_, window)| *window)
}

/// Parse LM Studio's `GET /api/v0/models`.
///
/// `loaded_context_length` is the length the model was actually loaded at —
/// the limit the user set in LM Studio, and what the server will enforce.
/// `max_context_length` is what the file supports, and is the best available
/// answer for a model that is not loaded yet.
pub fn parse_lm_studio_models(body: &serde_json::Value) -> DetectedWindows {
    let mut windows = DetectedWindows::new();
    let Some(entries) = body.get("data").and_then(|d| d.as_array()) else {
        return windows;
    };
    for entry in entries {
        let Some(id) = entry.get("id").and_then(|v| v.as_str()) else {
            continue;
        };
        let window = entry
            .get("loaded_context_length")
            .and_then(serde_json::Value::as_u64)
            .filter(|n| *n > 0)
            .or_else(|| {
                entry
                    .get("max_context_length")
                    .and_then(serde_json::Value::as_u64)
                    .filter(|n| *n > 0)
            });
        if let Some(window) = window {
            windows.insert(id.to_owned(), window);
        }
    }
    windows
}

/// Parse Ollama's `GET /api/ps`, whose `context_length` is the window the
/// running instance was loaded with (`num_ctx` / `OLLAMA_CONTEXT_LENGTH`) —
/// not the model's maximum.
pub fn parse_ollama_ps(body: &serde_json::Value) -> DetectedWindows {
    let mut windows = DetectedWindows::new();
    let Some(entries) = body.get("models").and_then(|m| m.as_array()) else {
        return windows;
    };
    for entry in entries {
        let Some(id) = entry
            .get("model")
            .and_then(|v| v.as_str())
            .or_else(|| entry.get("name").and_then(|v| v.as_str()))
        else {
            continue;
        };
        if let Some(window) = entry
            .get("context_length")
            .and_then(serde_json::Value::as_u64)
            .filter(|n| *n > 0)
        {
            windows.insert(id.to_owned(), window);
        }
    }
    windows
}

/// Parse Ollama's `POST /api/show`, used for a model that is not loaded.
///
/// The architecture-scoped `<arch>.context_length` in `model_info` is the
/// model's trained maximum. Ollama serves it at `num_ctx` unless configured
/// otherwise, so treat it as an upper bound, not a promise.
pub fn parse_ollama_show(body: &serde_json::Value) -> Option<u64> {
    let info = body.get("model_info")?.as_object()?;
    let arch = info.get("general.architecture").and_then(|v| v.as_str());
    if let Some(arch) = arch
        && let Some(window) = info
            .get(&format!("{arch}.context_length"))
            .and_then(serde_json::Value::as_u64)
            .filter(|n| *n > 0)
    {
        return Some(window);
    }
    // Fall back to any `*.context_length` — the architecture key is missing on
    // some older builds.
    info.iter()
        .filter(|(key, _)| key.ends_with(".context_length"))
        .find_map(|(_, value)| value.as_u64().filter(|n| *n > 0))
}

/// Fetch and parse one JSON body, mapping every failure to `None`. Probes are
/// advisory, so a down server must not surface as an error anywhere.
fn get_json(url: &str) -> Option<serde_json::Value> {
    let client = crate::http::shared_blocking_client();
    let response = client.get(url).timeout(PROBE_TIMEOUT).send().ok()?;
    if !response.status().is_success() {
        tracing::debug!(
            url,
            status = response.status().as_u16(),
            "local model probe failed"
        );
        return None;
    }
    response.json().ok()
}

fn post_json(url: &str, body: serde_json::Value) -> Option<serde_json::Value> {
    let client = crate::http::shared_blocking_client();
    let response = client
        .post(url)
        .timeout(PROBE_TIMEOUT)
        .json(&body)
        .send()
        .ok()?;
    if !response.status().is_success() {
        tracing::debug!(
            url,
            status = response.status().as_u16(),
            "local model probe failed"
        );
        return None;
    }
    response.json().ok()
}

struct CacheEntry<T> {
    value: T,
    fetched_at: Instant,
}

/// Catalog probes, keyed by provider and server root.
static PROBE_CACHE: LazyLock<Mutex<HashMap<String, CacheEntry<DetectedWindows>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Per-model fallback probes, keyed by provider, server root, and model.
/// Separate from [`PROBE_CACHE`] because it caches misses too — without that, a
/// catalog of Ollama models with none loaded would re-issue one `/api/show` per
/// model on every config load.
static MODEL_PROBE_CACHE: LazyLock<Mutex<HashMap<String, CacheEntry<Option<u64>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Read a live entry, or `None` when absent or expired.
fn cache_get<T: Clone>(cache: &Mutex<HashMap<String, CacheEntry<T>>>, key: &str) -> Option<T> {
    let guard = cache.lock().ok()?;
    let entry = guard.get(key)?;
    (entry.fetched_at.elapsed() < PROBE_TTL).then(|| entry.value.clone())
}

fn cache_put<T>(cache: &Mutex<HashMap<String, CacheEntry<T>>>, key: String, value: T) {
    if let Ok(mut guard) = cache.lock() {
        guard.insert(
            key,
            CacheEntry {
                value,
                fetched_at: Instant::now(),
            },
        );
    }
}

/// Drop every cached probe. Tests call this so one test's stub server does not
/// answer for the next.
pub fn clear_probe_cache() {
    if let Ok(mut cache) = PROBE_CACHE.lock() {
        cache.clear();
    }
    if let Ok(mut cache) = MODEL_PROBE_CACHE.lock() {
        cache.clear();
    }
}

/// Context windows for every model `base_url`'s server can report, cached for
/// [`PROBE_TTL`].
///
/// For Ollama this covers loaded models only — `/api/ps` is the only endpoint
/// that knows the window a model is actually being served at. Use
/// [`probe_model_context_window`] to also consult `/api/show` for one specific
/// model.
pub fn probe_context_windows(kind: LocalProviderKind, base_url: &str) -> DetectedWindows {
    let root = server_root(base_url);
    let cache_key = format!("{}|{root}", kind.id());
    if let Some(cached) = cache_get(&PROBE_CACHE, &cache_key) {
        return cached;
    }
    let windows = match kind {
        LocalProviderKind::LmStudio => get_json(&format!("{root}/api/v0/models"))
            .as_ref()
            .map(parse_lm_studio_models)
            .unwrap_or_default(),
        LocalProviderKind::Ollama => get_json(&format!("{root}/api/ps"))
            .as_ref()
            .map(parse_ollama_ps)
            .unwrap_or_default(),
    };
    tracing::debug!(
        provider = kind.id(),
        root,
        detected = windows.len(),
        "probed local server for context windows"
    );
    cache_put(&PROBE_CACHE, cache_key, windows.clone());
    windows
}

/// The context window `model` will actually be served at, or `None` when the
/// server cannot say.
///
/// Prefers the running instance's window, then — for Ollama, which can report a
/// window for a model it has not loaded — the model's maximum.
pub fn probe_model_context_window(
    kind: LocalProviderKind,
    base_url: &str,
    model: &str,
) -> Option<u64> {
    if let Some(window) = lookup_window(&probe_context_windows(kind, base_url), model) {
        return Some(window);
    }
    // LM Studio's catalog already covers every model it knows about, loaded or
    // not, so a miss there is final.
    if kind == LocalProviderKind::LmStudio {
        return None;
    }
    let root = server_root(base_url);
    let cache_key = format!("{}|{root}|show|{model}", kind.id());
    if let Some(cached) = cache_get(&MODEL_PROBE_CACHE, &cache_key) {
        return cached;
    }
    let window = post_json(
        &format!("{root}/api/show"),
        serde_json::json!({ "model": model }),
    )
    .as_ref()
    .and_then(parse_ollama_show);
    cache_put(&MODEL_PROBE_CACHE, cache_key, window);
    window
}

/// Which local server, if any, backs this model entry.
fn kind_for_entry(entry: &ModelEntry, cfg: &Config, key: &str) -> Option<LocalProviderKind> {
    let declared = cfg
        .config_models
        .get(key)
        .and_then(|m| m.model_provider.as_deref())
        .and_then(LocalProviderKind::from_provider_id);
    declared.or_else(|| LocalProviderKind::from_base_url(&entry.info.base_url))
}

/// Fill in detected context windows and scale the budgets that assume a
/// hosted-scale window.
///
/// Runs as the last layer of `resolve_model_list`, so everything it reads is
/// already merged. Only ever tightens, and only where the user left the value
/// unset:
///
/// * `context_window` — replaced by the detected value unless `[model.<key>]`
///   pinned one. Without this a local model inherits a 256K default and the
///   session overruns an 8K server mid-turn.
/// * `auto_compact_threshold_percent` — lowered per
///   [`xai_token_estimation::auto_compact_threshold_for_window`] so compaction
///   fires while a reply still fits. This is the per-model tier, so config and
///   env overrides continue to win.
/// * `use_concise` — enabled for small windows, trading the full system prompt
///   and toolset for room to work in.
/// * `api_backend` — set to Responses, which both servers implement and this
///   client targets. Only applies where nothing else chose a backend: a model
///   discovered from `/v1/models` carries the enum default rather than a
///   deliberate choice.
pub fn apply_local_provider_defaults(resolved: &mut IndexMap<String, ModelEntry>, cfg: &Config) {
    for (key, entry) in resolved.iter_mut() {
        let Some(kind) = kind_for_entry(entry, cfg, key) else {
            continue;
        };
        let pinned = cfg.config_models.get(key);
        if pinned.is_none_or(|m| m.api_backend.is_none())
            && entry.info.api_backend == ApiBackend::default()
        {
            entry.info.api_backend = ApiBackend::Responses;
        }
        if pinned.is_none_or(|m| m.context_window.is_none())
            && let Some(detected) =
                probe_model_context_window(kind, &entry.info.base_url, &entry.info.model)
            && let Some(detected) = std::num::NonZeroU64::new(detected)
        {
            if detected != entry.info.context_window {
                tracing::info!(
                    model_key = %key,
                    model = %entry.info.model,
                    provider = kind.id(),
                    from = entry.info.context_window.get(),
                    to = detected.get(),
                    "detected context window from {}",
                    kind.display_name(),
                );
            }
            entry.info.context_window = detected;
        }

        let window = entry.info.context_window.get();
        let configured = entry
            .info
            .auto_compact_threshold_percent
            .unwrap_or(xai_grok_compaction::DEFAULT_AUTO_COMPACT_THRESHOLD_PERCENT);
        let scaled = xai_token_estimation::auto_compact_threshold_for_window(window, configured);
        // Written only when it tightens, so a local model with a roomy window
        // keeps whatever tier would otherwise have supplied its threshold.
        if scaled < configured {
            tracing::debug!(
                model_key = %key,
                context_window = window,
                from = configured,
                to = scaled,
                "small context window; lowering auto-compact threshold",
            );
            entry.info.auto_compact_threshold_percent = Some(scaled);
        }
        if xai_token_estimation::is_small_context_window(window)
            && pinned.is_none_or(|m| m.use_concise.is_none())
            && !entry.info.use_concise
        {
            tracing::debug!(
                model_key = %key,
                context_window = window,
                "small context window; enabling concise mode",
            );
            entry.info.use_concise = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::config::{ConfigModelOverride, ModelInfo};

    #[test]
    fn provider_ids_tolerate_spelling() {
        for id in [
            "lmstudio",
            "lm-studio",
            "lm_studio",
            "LM Studio",
            "LMStudio",
        ] {
            assert_eq!(
                LocalProviderKind::from_provider_id(id),
                Some(LocalProviderKind::LmStudio),
                "{id} should resolve to LM Studio"
            );
        }
        assert_eq!(
            LocalProviderKind::from_provider_id("ollama"),
            Some(LocalProviderKind::Ollama)
        );
        assert_eq!(LocalProviderKind::from_provider_id("openai"), None);
        assert_eq!(LocalProviderKind::from_provider_id(""), None);
    }

    #[test]
    fn base_urls_are_recognized_by_port_or_host() {
        assert_eq!(
            LocalProviderKind::from_base_url("http://localhost:1234/v1"),
            Some(LocalProviderKind::LmStudio)
        );
        assert_eq!(
            LocalProviderKind::from_base_url("http://127.0.0.1:11434/v1"),
            Some(LocalProviderKind::Ollama)
        );
        assert_eq!(
            LocalProviderKind::from_base_url("http://ollama.internal:8080/v1"),
            Some(LocalProviderKind::Ollama)
        );
        // A loopback server on some other port is not assumed to be either.
        assert_eq!(
            LocalProviderKind::from_base_url("http://localhost:8000/v1"),
            None
        );
        // Nor is a remote host on a coincidental port.
        assert_eq!(
            LocalProviderKind::from_base_url("https://api.meta.ai/v1"),
            None
        );
    }

    #[test]
    fn server_root_strips_openai_suffix() {
        assert_eq!(
            server_root("http://localhost:1234/v1"),
            "http://localhost:1234"
        );
        assert_eq!(
            server_root("http://localhost:1234/v1/"),
            "http://localhost:1234"
        );
        assert_eq!(
            server_root("http://localhost:11434"),
            "http://localhost:11434"
        );
        // Only the trailing /v1 goes — a mounted path is preserved.
        assert_eq!(server_root("http://box:1234/llm/v1"), "http://box:1234/llm");
    }

    #[test]
    fn lm_studio_prefers_the_loaded_length() {
        let body = serde_json::json!({
            "data": [
                {
                    "id": "qwen/qwen3-coder-30b",
                    "state": "loaded",
                    "max_context_length": 262_144,
                    "loaded_context_length": 16_384
                },
                {
                    "id": "mistral-7b",
                    "state": "not-loaded",
                    "max_context_length": 32_768
                }
            ]
        });
        let windows = parse_lm_studio_models(&body);
        // The set limit wins over what the file could support.
        assert_eq!(windows.get("qwen/qwen3-coder-30b"), Some(&16_384));
        // Not loaded: its maximum is the only answer available.
        assert_eq!(windows.get("mistral-7b"), Some(&32_768));
    }

    #[test]
    fn lm_studio_skips_unusable_entries() {
        let body = serde_json::json!({
            "data": [
                { "id": "no-window", "state": "not-loaded" },
                { "id": "zero-window", "max_context_length": 0 },
                { "state": "loaded", "max_context_length": 4096 },
                { "id": "zero-loaded", "loaded_context_length": 0, "max_context_length": 8192 }
            ]
        });
        let windows = parse_lm_studio_models(&body);
        assert!(!windows.contains_key("no-window"));
        assert!(!windows.contains_key("zero-window"));
        // A zero `loaded_context_length` falls through to the maximum.
        assert_eq!(windows.get("zero-loaded"), Some(&8192));
        assert_eq!(windows.len(), 1);
    }

    #[test]
    fn lm_studio_tolerates_unexpected_bodies() {
        assert!(parse_lm_studio_models(&serde_json::json!({})).is_empty());
        assert!(parse_lm_studio_models(&serde_json::json!({ "data": "nope" })).is_empty());
        assert!(parse_lm_studio_models(&serde_json::json!([])).is_empty());
    }

    #[test]
    fn ollama_ps_reads_the_running_window() {
        let body = serde_json::json!({
            "models": [
                {
                    "name": "qwen3-coder:30b",
                    "model": "qwen3-coder:30b",
                    "size": 20_000_000_000_u64,
                    "context_length": 8_192
                },
                { "name": "llama3:latest", "model": "llama3:latest" }
            ]
        });
        let windows = parse_ollama_ps(&body);
        assert_eq!(windows.get("qwen3-coder:30b"), Some(&8_192));
        // No context_length reported: nothing to claim.
        assert!(!windows.contains_key("llama3:latest"));
    }

    #[test]
    fn ollama_show_reads_the_architecture_key() {
        let body = serde_json::json!({
            "model_info": {
                "general.architecture": "qwen3",
                "qwen3.context_length": 262_144,
                "qwen3.embedding_length": 5_120
            }
        });
        assert_eq!(parse_ollama_show(&body), Some(262_144));
    }

    #[test]
    fn ollama_show_falls_back_without_architecture() {
        let body = serde_json::json!({
            "model_info": { "llama.context_length": 131_072 }
        });
        assert_eq!(parse_ollama_show(&body), Some(131_072));
        assert_eq!(
            parse_ollama_show(&serde_json::json!({ "model_info": {} })),
            None
        );
        assert_eq!(parse_ollama_show(&serde_json::json!({})), None);
    }

    #[test]
    fn lookup_tolerates_the_latest_tag() {
        let mut windows = DetectedWindows::new();
        windows.insert("llama3:latest".to_owned(), 8_192);
        assert_eq!(lookup_window(&windows, "llama3:latest"), Some(8_192));
        assert_eq!(lookup_window(&windows, "llama3"), Some(8_192));
        assert_eq!(lookup_window(&windows, "llama2"), None);
    }

    #[test]
    fn preset_targets_the_responses_api() {
        for kind in [LocalProviderKind::LmStudio, LocalProviderKind::Ollama] {
            let preset = preset(kind);
            assert_eq!(preset.api_backend, Some(ApiBackend::Responses));
            assert_eq!(preset.base_url.as_deref(), Some(kind.default_base_url()));
            // Detection owns the window; a preset guess would outrank it.
            assert_eq!(preset.context_window, None);
        }
    }

    #[test]
    fn injecting_presets_leaves_user_values_alone() {
        let mut providers = IndexMap::new();
        providers.insert(
            "ollama".to_owned(),
            ModelProviderConfig {
                base_url: Some("http://gpu-box:11434/v1".to_owned()),
                ..Default::default()
            },
        );
        inject_presets(&mut providers, ["ollama".to_owned(), "lmstudio".to_owned()]);

        let ollama = &providers["ollama"];
        assert_eq!(ollama.base_url.as_deref(), Some("http://gpu-box:11434/v1"));
        // Unset fields still come from the preset.
        assert_eq!(ollama.api_backend, Some(ApiBackend::Responses));
        assert_eq!(
            ollama.api_key.as_deref(),
            Some(LocalProviderKind::Ollama.placeholder_api_key())
        );
        assert_eq!(
            providers["lmstudio"].base_url.as_deref(),
            Some(LocalProviderKind::LmStudio.default_base_url())
        );
    }

    #[test]
    fn injecting_presets_does_not_shadow_a_real_credential() {
        let mut providers = IndexMap::new();
        providers.insert(
            "lmstudio".to_owned(),
            ModelProviderConfig {
                api_key: Some("sk-real-key".to_owned()),
                ..Default::default()
            },
        );
        inject_presets(&mut providers, ["lmstudio".to_owned()]);
        assert_eq!(
            providers["lmstudio"].api_key.as_deref(),
            Some("sk-real-key")
        );
        assert!(providers["lmstudio"].env_key.is_none());
    }

    #[test]
    fn injecting_presets_ignores_unknown_providers() {
        let mut providers = IndexMap::new();
        inject_presets(&mut providers, ["vllm".to_owned()]);
        assert!(providers.is_empty());
    }

    // ── Small-window adjustments ────────────────────────────────────────
    //
    // These drive `apply_local_provider_defaults` without a server. The model
    // declares `model_provider = "lmstudio"` — which is what selects the
    // provider — while its `base_url` points at a closed port, so the probe
    // fails fast and deterministically. Pointing at the real default port would
    // make the result depend on whether the machine running the tests happens
    // to have LM Studio up.
    const CLOSED_PORT_URL: &str = "http://127.0.0.1:9/v1";

    /// Build a one-model catalog plus the config that declares it.
    fn fixture(
        window: u64,
        adjust: impl FnOnce(&mut ConfigModelOverride),
    ) -> (IndexMap<String, ModelEntry>, Config) {
        let mut info = ModelInfo::fallback("local-model");
        info.base_url = CLOSED_PORT_URL.to_owned();
        info.context_window = std::num::NonZeroU64::new(window).expect("non-zero window");
        let entry = ModelEntry {
            info,
            api_key: None,
            env_key: None,
            auth_provider: None,
            api_base_url: None,
        };

        let mut model_override = ConfigModelOverride {
            model_provider: Some("lmstudio".to_owned()),
            ..Default::default()
        };
        adjust(&mut model_override);

        let mut cfg = Config::default();
        cfg.config_models.insert("local".to_owned(), model_override);
        (IndexMap::from([("local".to_owned(), entry)]), cfg)
    }

    fn apply(window: u64, adjust: impl FnOnce(&mut ConfigModelOverride)) -> ModelEntry {
        clear_probe_cache();
        let (mut resolved, cfg) = fixture(window, adjust);
        apply_local_provider_defaults(&mut resolved, &cfg);
        resolved.shift_remove("local").expect("entry survives")
    }

    #[test]
    fn small_window_tightens_threshold_and_enables_concise() {
        let entry = apply(8_192, |_| {});
        assert_eq!(entry.info.auto_compact_threshold_percent, Some(75));
        assert!(entry.info.use_concise);
    }

    /// A local model with a roomy window must come out untouched — no
    /// materialized threshold that would shadow the tier that owns it.
    #[test]
    fn large_window_is_left_alone() {
        let entry = apply(200_000, |_| {});
        assert_eq!(entry.info.auto_compact_threshold_percent, None);
        assert!(!entry.info.use_concise);
    }

    /// Scaling only tightens: a stricter threshold already in place survives.
    #[test]
    fn a_stricter_threshold_is_not_loosened() {
        clear_probe_cache();
        let (mut resolved, cfg) = fixture(8_192, |_| {});
        resolved["local"].info.auto_compact_threshold_percent = Some(50);
        apply_local_provider_defaults(&mut resolved, &cfg);
        assert_eq!(
            resolved["local"].info.auto_compact_threshold_percent,
            Some(50)
        );
    }

    #[test]
    fn explicit_use_concise_false_is_respected() {
        let entry = apply(8_192, |m| m.use_concise = Some(false));
        assert!(!entry.info.use_concise);
        // The threshold is a separate knob and still tightens.
        assert_eq!(entry.info.auto_compact_threshold_percent, Some(75));
    }

    /// A pinned window is never replaced by a probe — and with the server
    /// unreachable here, it is also never lost.
    #[test]
    fn pinned_context_window_survives() {
        let entry = apply(8_192, |m| m.context_window = Some(8_192));
        assert_eq!(entry.info.context_window.get(), 8_192);
    }

    #[test]
    fn backend_defaults_to_responses() {
        let entry = apply(8_192, |_| {});
        assert_eq!(entry.info.api_backend, ApiBackend::Responses);
    }

    #[test]
    fn an_explicit_backend_is_respected() {
        let entry = apply(8_192, |m| m.api_backend = Some(ApiBackend::ChatCompletions));
        assert_eq!(entry.info.api_backend, ApiBackend::ChatCompletions);
    }

    /// Models that are not local must not be touched — in particular, they
    /// must never trigger a probe.
    #[test]
    fn hosted_models_are_ignored() {
        clear_probe_cache();
        let mut info = ModelInfo::fallback("hosted");
        info.base_url = "https://api.meta.ai/v1".to_owned();
        info.context_window = std::num::NonZeroU64::new(8_192).expect("non-zero");
        let mut resolved = IndexMap::from([(
            "hosted".to_owned(),
            ModelEntry {
                info,
                api_key: None,
                env_key: None,
                auth_provider: None,
                api_base_url: None,
            },
        )]);

        apply_local_provider_defaults(&mut resolved, &Config::default());

        let entry = &resolved["hosted"];
        assert_eq!(entry.info.auto_compact_threshold_percent, None);
        assert!(!entry.info.use_concise);
        assert_eq!(entry.info.api_backend, ApiBackend::default());
    }
}
