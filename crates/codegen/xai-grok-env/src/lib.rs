//! Backend environment presets for the Guac Build crate family: endpoint URL
//! defaults, environment selection, and env-var test support.
//!
//! Public builds expose production endpoints. Values resolve as a `GROK_*`
//! env-var override when set, else the compiled production default.
//!
//! # First-party endpoints
//! This fork talks to one first-party inference host, the OpenAI-compatible
//! Meta Model API. There is no vendor-operated CLI proxy in front of it: the
//! client authenticates with an API key (or a configured OIDC session bearer)
//! and speaks plain OpenAI wire format. [`FIRST_PARTY_API_HOST`] is the single
//! source of truth for "is this endpoint ours" decisions — credential-refusal
//! kill switches and, more importantly, the check that keeps a session bearer
//! from being attached to a third-party `base_url`.
/// The endpoint set for one backend environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrokBuildEndpoints {
    pub api_base_url: &'static str,
    /// TODO: dies with the `grok.com/code` web-frontend relay bridge.
    pub relay_ws_url: &'static str,
    /// TODO: dies with the relay bridge.
    pub ws_origin: &'static str,
}
/// Host of the first-party inference API. Bare host, no scheme or path, so
/// trust checks can compare it against a parsed URL's `host_str()`.
pub const FIRST_PARTY_API_HOST: &str = "api.meta.ai";
const PRODUCTION_ENDPOINTS: GrokBuildEndpoints = GrokBuildEndpoints {
    api_base_url: "https://api.meta.ai/v1",
    relay_ws_url: "wss://code.grok.com/ws/code-agent",
    ws_origin: "https://grok.com",
};
/// Default base URL for the first-party inference API. Every other
/// `*_BASE_URL_DEFAULT` in the tree derives from this one so a repoint cannot
/// leave a stale duplicate behind.
pub const PROD_API_BASE_URL: &str = PRODUCTION_ENDPOINTS.api_base_url;
pub const PROD_RELAY_WS_URL: &str = PRODUCTION_ENDPOINTS.relay_ws_url;
pub const PROD_WS_ORIGIN: &str = PRODUCTION_ENDPOINTS.ws_origin;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GrokBuildEnvironment {
    #[default]
    Production,
}
impl GrokBuildEnvironment {
    pub fn from_flags(_dev: bool, _staging: bool) -> Self {
        GrokBuildEnvironment::Production
    }
    /// Indicator string for display; `None` for Production.
    pub fn indicator(&self) -> Option<&'static str> {
        match self {
            GrokBuildEnvironment::Production => None,
        }
    }
    pub fn is_production(&self) -> bool {
        matches!(self, GrokBuildEnvironment::Production)
    }
    fn env_prefix(&self) -> &'static str {
        match self {
            GrokBuildEnvironment::Production => "GROK_PRODUCTION",
        }
    }
    /// Compiled endpoint set for this environment (production by default).
    pub fn endpoints(&self) -> GrokBuildEndpoints {
        match self {
            GrokBuildEnvironment::Production => PRODUCTION_ENDPOINTS,
        }
    }
    /// Env-var override when set, else the compiled endpoint.
    fn resolve(&self, var_suffix: &str, compiled: &'static str) -> String {
        std::env::var(format!("{}{var_suffix}", self.env_prefix()))
            .unwrap_or_else(|_| compiled.to_string())
    }
    /// Base URL of the first-party inference API.
    pub fn api_base_url(&self) -> String {
        self.resolve("_API_BASE_URL", self.endpoints().api_base_url)
    }
    /// The relay WebSocket URL (`grok.com/code` web frontend driving a local
    /// agent). TODO: dies with the relay bridge.
    pub fn relay_ws_url(&self) -> String {
        self.resolve("_WS_URL", self.endpoints().relay_ws_url)
    }
    pub fn ws_origin(&self) -> String {
        self.resolve("_WS_ORIGIN", self.endpoints().ws_origin)
    }
}
impl std::fmt::Display for GrokBuildEnvironment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GrokBuildEnvironment::Production => write!(f, "production"),
        }
    }
}
/// Serializes env-var mutation across tests; `std::env` is process-global.
#[cfg(any(test, feature = "test-support"))]
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
#[cfg(any(test, feature = "test-support"))]
fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}
/// RAII env-var override for tests: constructors snapshot the prior value
/// under [`ENV_LOCK`], `Drop` restores it, panics included.
#[cfg(any(test, feature = "test-support"))]
pub struct EnvVarGuard {
    key: &'static str,
    prev: Option<String>,
    _lock: std::sync::MutexGuard<'static, ()>,
}
#[cfg(any(test, feature = "test-support"))]
impl EnvVarGuard {
    pub fn set(key: &'static str, value: &str) -> Self {
        let lock = env_lock();
        let prev = std::env::var(key).ok();
        unsafe { std::env::set_var(key, value) };
        Self {
            key,
            prev,
            _lock: lock,
        }
    }
    pub fn remove(key: &'static str) -> Self {
        let lock = env_lock();
        let prev = std::env::var(key).ok();
        unsafe { std::env::remove_var(key) };
        Self {
            key,
            prev,
            _lock: lock,
        }
    }
    /// Update the value while still holding the env lock.
    pub fn set_value(&self, value: &str) {
        unsafe { std::env::set_var(self.key, value) };
    }
}
#[cfg(any(test, feature = "test-support"))]
impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match self.prev.take() {
            Some(prev) => unsafe { std::env::set_var(self.key, prev) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    /// The env-var prefixes are an operator interface; do not rename.
    #[test]
    fn test_env_prefix() {
        assert_eq!(
            GrokBuildEnvironment::Production.env_prefix(),
            "GROK_PRODUCTION"
        );
    }
    #[test]
    fn env_var_guard_set_value_updates_then_restores_on_drop() {
        const KEY: &str = "XAI_GROK_ENV_VAR_GUARD_SET_VALUE_PROBE";
        let before = std::env::var(KEY).ok();
        {
            let guard = EnvVarGuard::set(KEY, "initial");
            assert_eq!(std::env::var(KEY).ok().as_deref(), Some("initial"));
            guard.set_value("updated");
            assert_eq!(
                std::env::var(KEY).ok().as_deref(),
                Some("updated"),
                "set_value must update the env var while the guard is live"
            );
        }
        assert_eq!(
            std::env::var(KEY).ok(),
            before,
            "Drop must restore the pre-guard snapshot (was {before:?})"
        );
    }
    /// The compiled default and the host used for first-party trust checks must
    /// describe the same endpoint. If they drift, a session bearer stops being
    /// attached to the fork's own API (or starts being attached to someone
    /// else's), which is exactly the bug the rebrand introduced once already.
    #[test]
    fn first_party_host_matches_compiled_api_base_url() {
        let parsed = url::Url::parse(PROD_API_BASE_URL).expect("compiled base URL must parse");
        assert_eq!(parsed.host_str(), Some(FIRST_PARTY_API_HOST));
        assert_eq!(parsed.scheme(), "https");
    }
    #[test]
    fn test_from_flags() {
        assert_eq!(
            GrokBuildEnvironment::from_flags(false, false),
            GrokBuildEnvironment::Production
        );
    }
}
