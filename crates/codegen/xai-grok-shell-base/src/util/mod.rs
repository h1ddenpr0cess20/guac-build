pub mod changelog;
pub mod event_id;
pub mod grok_home;
pub mod secure_file;
pub mod tips;
pub mod uname;
pub use xai_grok_shared::clipboard;
pub use xai_grok_shared::stderr::{stderr_lock, with_locked_stderr};
/// Generate a pseudo-random f64 in [0.0, 1.0).
///
/// Uses `RandomState::new()` which is OS-seeded (via `getrandom`) on each
/// instantiation, producing a unique hasher state per call. A fixed sentinel
/// is hashed to extract the random bits — the entropy comes entirely from
/// the OS-seeded `RandomState`, not from any clock source.
///
/// # Precision
/// The result uses all 53 bits of `f64` mantissa for a uniform distribution
/// over `[0.0, 1.0)`. We shift the 64-bit hash right by 11 bits to get a
/// 53-bit integer, then divide by `2^53`. This avoids the subtle bias that
/// occurs when casting a full `u64` to `f64` (which has only 52 bits of
/// mantissa, causing multiple `u64` values to map to the same `f64` for
/// values > 2^52).
///
/// Not cryptographically secure — suitable for sampling and feature
/// rollouts, not for security-sensitive randomness.
pub fn random_f64() -> f64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let random_state = RandomState::new();
    let mut hasher = random_state.build_hasher();
    hasher.write_u64(0x517cc1b727220a95);
    (hasher.finish() >> 11) as f64 / (1u64 << 53) as f64
}
/// Probabilistic sampling. Returns `true` with probability `rate` (0.0–1.0).
pub fn probabilistic_sample(rate: f64) -> bool {
    random_f64() < rate
}
/// True for the first-party inference API ([`crate::env::FIRST_PARTY_API_HOST`]
/// and its subdomains).
///
/// `disable_api_key_auth` refuses keys only for these; every other host is BYOK
/// and exempt. Safe against invalid URLs and suffix attacks
/// (`evil-api.meta.ai.example`, `prefixmeta.ai`).
///
/// Scheme-agnostic so credential *refusal* fails closed. To decide where to
/// *attach* a credential, use [`is_first_party_bearer_url`].
pub fn is_first_party_api_url(url: &str) -> bool {
    is_first_party_api_url_impl(url, false)
}
/// Like [`is_first_party_api_url`], but requires `https` on every arm, so a
/// session bearer is never attached to a cleartext endpoint, including loopback
/// (a co-located process could otherwise read a token sent to `http://localhost`).
pub fn is_first_party_bearer_url(url: &str) -> bool {
    is_first_party_api_url_impl(url, true)
}
fn is_first_party_api_url_impl(url: &str, require_https: bool) -> bool {
    let Ok(parsed) = reqwest::Url::parse(url) else {
        return false;
    };
    if require_https {
        if parsed.scheme() != "https" {
            return false;
        }
        if is_loopback_host(&parsed) {
            return false;
        }
    }
    let host = parsed.host_str().unwrap_or_default().to_ascii_lowercase();
    let apex = crate::env::FIRST_PARTY_API_HOST;
    // Match the apex exactly, or a subdomain of it. The leading dot is what
    // stops `evil-api.meta.ai.example` and `prefixapi.meta.ai` from matching.
    host == apex || host.ends_with(&format!(".{apex}"))
}
/// True when the endpoint can be expected to serve the first-party API
/// *extensions* — `/models-v2` metadata, `/settings` — on top of plain
/// OpenAI-compatible chat: the first-party host, or any loopback address.
///
/// Loopback counts because a local mock server (the idle-resume tests) or a
/// user-run reverse proxy in front of the real API is still that API. This is
/// deliberately **not** the predicate for credential decisions: attaching a
/// bearer to loopback would leak it to any co-located process, and refusing an
/// API key on loopback would break LM Studio and Ollama. Use
/// [`is_first_party_bearer_url`] and [`is_first_party_api_url`] for those.
pub fn serves_first_party_api_extensions(url: &str) -> bool {
    if is_first_party_api_url(url) {
        return true;
    }
    reqwest::Url::parse(url).is_ok_and(|u| is_loopback_host(&u))
}
fn is_loopback_host(parsed: &reqwest::Url) -> bool {
    match parsed.host() {
        Some(url::Host::Domain(host)) => host == "localhost",
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        None => false,
    }
}
/// Truncate a string to at most `max_chars` characters.
/// Slices at char boundaries so multi-byte UTF-8 never panics.
pub fn truncate(s: &str, max_chars: usize) -> &str {
    if s.len() <= max_chars {
        return s;
    }
    let end = s
        .char_indices()
        .nth(max_chars)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    &s[..end]
}
/// Check if a process is still alive.
///
/// - Unix: `kill(pid, 0)` via `nix`. True if the process exists (even
///   under a different UID); false only on ESRCH.
/// - Windows: `OpenProcess(SYNCHRONIZE)` + `WaitForSingleObject(0)`. True
///   while running; false on exit, absence, or open failure.
#[cfg(unix)]
pub fn is_process_alive(pid: u32) -> bool {
    use nix::errno::Errno;
    use nix::sys::signal::kill;
    use nix::unistd::Pid;
    match kill(Pid::from_raw(pid as i32), None) {
        Ok(()) => true,
        Err(Errno::ESRCH) => false,
        Err(_) => true,
    }
}
#[cfg(windows)]
pub fn is_process_alive(pid: u32) -> bool {
    use windows::Win32::Foundation::{CloseHandle, WAIT_TIMEOUT};
    use windows::Win32::System::Threading::{
        OpenProcess, PROCESS_SYNCHRONIZE, WaitForSingleObject,
    };
    let Ok(handle) = (unsafe { OpenProcess(PROCESS_SYNCHRONIZE, false, pid) }) else {
        return false;
    };
    let wait_result = unsafe { WaitForSingleObject(handle, 0) };
    let _ = unsafe { CloseHandle(handle) };
    wait_result == WAIT_TIMEOUT
}
/// Terminate a process by PID. Idempotent: already-dead is `Ok`.
///
/// - Unix: `SIGTERM` via `nix::sys::signal::kill`; ESRCH maps to `Ok`.
/// - Windows: `OpenProcess(PROCESS_TERMINATE)` + `TerminateProcess`;
///   ERROR_INVALID_PARAMETER (Windows' "no such process") maps to `Ok`.
pub fn kill_process_by_pid(pid: u32) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use nix::errno::Errno;
        use nix::sys::signal::{Signal, kill};
        use nix::unistd::Pid;
        match kill(Pid::from_raw(pid as i32), Signal::SIGTERM) {
            Ok(()) | Err(Errno::ESRCH) => Ok(()),
            Err(e) => Err(std::io::Error::from_raw_os_error(e as i32)),
        }
    }
    #[cfg(windows)]
    {
        use windows::Win32::Foundation::{CloseHandle, ERROR_INVALID_PARAMETER};
        use windows::Win32::System::Threading::{OpenProcess, PROCESS_TERMINATE, TerminateProcess};
        use windows::core::HRESULT;
        let no_such_process = HRESULT::from_win32(ERROR_INVALID_PARAMETER.0);
        let handle = match unsafe { OpenProcess(PROCESS_TERMINATE, false, pid) } {
            Ok(h) => h,
            Err(e) if e.code() == no_such_process => return Ok(()),
            Err(e) => {
                return Err(std::io::Error::other(format!("OpenProcess({pid}): {e}")));
            }
        };
        let terminate = unsafe { TerminateProcess(handle, 0) };
        let _ = unsafe { CloseHandle(handle) };
        terminate.map_err(|e| std::io::Error::other(format!("TerminateProcess({pid}): {e}")))
    }
}
/// True if a process's command line / image path identifies one of our own
/// processes. Matches the current `guac` binary name and the legacy `grok`
/// name so leaders spawned by an older build (or a compatibility `$GROK_HOME`
/// install) are still recognized after the rebrand.
///
/// Pure and case-sensitive: Linux passes the raw `/proc/<pid>/cmdline`, Windows
/// passes the already-lowercased image path (both binary names are lowercase).
fn cmdline_identifies_our_process(text: &str) -> bool {
    text.contains("guac") || text.contains("grok")
}
/// True if `pid` is a guac process; pairs with [`kill_process_by_pid`] to avoid killing a recycled PID.
/// Best-effort on macOS/BSD (liveness-only via `kill -0`), exact on Linux (/proc cmdline) and Windows (image path).
pub fn is_grok_process(pid: u32) -> bool {
    #[cfg(target_os = "linux")]
    {
        let cmdline_path = format!("/proc/{pid}/cmdline");
        match std::fs::read(&cmdline_path) {
            Ok(data) => cmdline_identifies_our_process(&String::from_utf8_lossy(&data)),
            Err(_) => false,
        }
    }
    #[cfg(windows)]
    {
        use windows::Win32::Foundation::CloseHandle;
        use windows::Win32::System::Threading::{
            OpenProcess, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION,
            QueryFullProcessImageNameW,
        };
        use windows::core::PWSTR;
        let Ok(handle) = (unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) })
        else {
            return false;
        };
        let mut buf: Vec<u16> = vec![0; 1024];
        let mut size: u32 = buf.len() as u32;
        let result = unsafe {
            QueryFullProcessImageNameW(
                handle,
                PROCESS_NAME_WIN32,
                PWSTR(buf.as_mut_ptr()),
                &mut size,
            )
        };
        let _ = unsafe { CloseHandle(handle) };
        if result.is_err() {
            return false;
        }
        let image = String::from_utf16_lossy(&buf[..size as usize]).to_ascii_lowercase();
        cmdline_identifies_our_process(&image)
    }
    #[cfg(all(not(target_os = "linux"), not(windows)))]
    {
        let mut cmd = std::process::Command::new("kill");
        cmd.args(["-0", &pid.to_string()])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        xai_tty_utils::detach_std_command(&mut cmd);
        cmd.status().is_ok_and(|s| s.success())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_is_first_party_api_url() {
        assert!(is_first_party_api_url("https://api.meta.ai/v1"));
        assert!(is_first_party_api_url(
            "https://api.meta.ai/v1/chat/completions"
        ));
        // A path outside the compiled `/v1` prefix is still the same operator.
        assert!(is_first_party_api_url("https://api.meta.ai/"));
        assert!(!is_first_party_api_url("https://api.openai.com/v1"));
        assert!(!is_first_party_api_url("https://api.anthropic.com/v1"));
        assert!(!is_first_party_api_url(
            "https://generativelanguage.googleapis.com"
        ));
        // Suffix-confusion attacks.
        assert!(!is_first_party_api_url(
            "https://api.meta.ai.evil.example/v1"
        ));
        assert!(!is_first_party_api_url(
            "https://evil-api.meta.ai.attacker.com/v1"
        ));
        assert!(!is_first_party_api_url("https://prefixapi.meta.ai/v1"));
        assert!(!is_first_party_api_url("not-a-url"));
        assert!(!is_first_party_api_url(""));
        // Scheme-agnostic, so credential *refusal* fails closed.
        assert!(is_first_party_api_url("http://api.meta.ai/v1"));
        // A local model server is the user's own machine, not first-party: the
        // `disable_api_key_auth` kill switch must not strip an LM Studio or
        // Ollama key.
        assert!(!is_first_party_api_url("http://localhost:11434/v1"));
    }
    #[test]
    fn test_is_first_party_bearer_url() {
        assert!(is_first_party_bearer_url("https://api.meta.ai/v1"));
        assert!(!is_first_party_bearer_url("http://api.meta.ai/v1"));
        assert!(!is_first_party_bearer_url("http://localhost:11434/v1"));
        {
            assert!(!is_first_party_bearer_url("https://localhost:11434/v1"));
            assert!(!is_first_party_bearer_url("https://127.0.0.2:11434/v1"));
            assert!(!is_first_party_bearer_url("https://[::1]:11434/v1"));
        }
        assert!(is_first_party_bearer_url("https://API.META.AI/v1"));
        // userinfo trick: the real host is `attacker.example`.
        assert!(!is_first_party_bearer_url(
            "https://api.meta.ai@attacker.example/v1"
        ));
        // Cyrillic homoglyph in the apex label.
        assert!(!is_first_party_bearer_url("https://api.mеta.ai/v1"));
    }
    /// Loopback serves the extended surface (local mock servers, a user-run
    /// reverse proxy) but must never be treated as first-party for credentials.
    #[test]
    fn serves_first_party_api_extensions_accepts_loopback_but_credentials_do_not() {
        for local in [
            "http://localhost:1234/v1",
            "http://127.0.0.1:8080/v1",
            "https://[::1]:9000/v1",
        ] {
            assert!(
                serves_first_party_api_extensions(local),
                "{local} should serve extensions"
            );
            assert!(
                !is_first_party_bearer_url(local),
                "{local} must never receive a session bearer"
            );
            assert!(
                !is_first_party_api_url(local),
                "{local} must stay exempt from the api-key kill switch"
            );
        }
        assert!(serves_first_party_api_extensions(
            crate::env::PROD_API_BASE_URL
        ));
        assert!(!serves_first_party_api_extensions(
            "https://api.openai.com/v1"
        ));
        assert!(!serves_first_party_api_extensions("not-a-url"));
    }
    /// The bearer predicate must accept the compiled default, or the fork cannot
    /// authenticate to its own API. This is the regression the rebrand shipped:
    /// the default moved to `api.meta.ai` while the trust set still named
    /// `*.x.ai`, so the session bearer silently stopped being attached.
    #[test]
    fn compiled_default_base_url_is_trusted_for_bearer() {
        assert!(is_first_party_bearer_url(crate::env::PROD_API_BASE_URL));
        assert!(is_first_party_api_url(crate::env::PROD_API_BASE_URL));
    }
    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 5), "hello");
        assert_eq!(truncate("hello world", 5), "hello");
        assert_eq!(truncate("abc🎉🎉def", 5), "abc🎉🎉");
    }
    #[test]
    fn is_process_alive_current_process() {
        assert!(is_process_alive(std::process::id()));
    }
    #[test]
    fn is_process_alive_dead_pid() {
        assert!(!is_process_alive(4_000_000_000));
    }
    #[cfg(unix)]
    #[test]
    fn is_process_alive_init_process() {
        assert!(is_process_alive(1));
    }
    #[test]
    fn kill_process_by_pid_already_dead_is_ok() {
        assert!(kill_process_by_pid(4_000_000_000).is_ok());
    }
    #[cfg(unix)]
    #[test]
    fn kill_process_by_pid_terminates_live_child() {
        let mut child = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn sleep");
        let pid = child.id();
        kill_process_by_pid(pid).expect("kill should succeed");
        let status = child.wait().expect("wait child");
        assert!(
            !status.success(),
            "sleep was terminated, not exited cleanly"
        );
    }
    #[test]
    fn is_grok_process_self_true_impossible_pid_false() {
        assert!(is_grok_process(std::process::id()));
        assert!(!is_grok_process(u32::MAX));
    }
    #[test]
    fn cmdline_identifies_guac_and_legacy_grok_but_not_others() {
        // Regression guard for the rebrand: the shipped binary is `guac`, so a
        // real leader's cmdline/image never contains "grok". Before the fix,
        // `guac leader kill` matched only "grok" and skipped every live leader
        // (deleting its lock/socket as "stale"). The self-recognition test above
        // is masked because the test binary name contains "grok", so guard the
        // matcher directly.
        assert!(cmdline_identifies_our_process("/home/u/.guac/bin/guac"));
        assert!(cmdline_identifies_our_process(
            "target/debug/guac\0agent\0leader"
        ));
        // Legacy name still recognized (older installs / $GROK_HOME compat).
        assert!(cmdline_identifies_our_process("/home/u/.grok/bin/grok"));
        // Unrelated processes must not match.
        assert!(!cmdline_identifies_our_process(
            "/usr/bin/python3\0script.py"
        ));
        assert!(!cmdline_identifies_our_process("/bin/sleep\0300"));
        assert!(!cmdline_identifies_our_process(""));
    }
}
