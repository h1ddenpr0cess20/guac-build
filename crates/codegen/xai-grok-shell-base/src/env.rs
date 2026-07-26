//! GrokBuildEnvironment configuration for the shell crate family.
//!
//! The environment presets (per-environment endpoint URLs, `EnvVarGuard`) live
//! in the [`xai_grok_env`] leaf crate so sibling crates (telemetry, tools,
//! workspace) can share them without depending on this crate. This module
//! re-exports them.
#[cfg(any(test, feature = "test-support"))]
pub use xai_grok_env::EnvVarGuard;
pub use xai_grok_env::{
    FIRST_PARTY_API_HOST, GrokBuildEnvironment, PROD_API_BASE_URL, PROD_RELAY_WS_URL,
    PROD_WS_ORIGIN,
};
