//! Local session logging for Guac Build.
//!
//! This crate used to be the outbound telemetry engine: product events posted
//! to xAI's events endpoint, a Mixpanel client, a Sentry reporter, and two
//! OpenTelemetry exporters. All of it has been removed -- nothing here opens a
//! socket. What remains writes to `~/.guac/logs/` and never leaves the machine:
//!
//! - [`unified_log`] -- the structured session log
//! - [`debug_log`] -- the `--debug` firehose
//! - [`sampling_log`], [`hooks_log`], [`memory_log`] -- opt-in per-subsystem
//!   tracing layers
//! - [`instrumentation`] -- tracing-subscriber setup and the panic hook
//!
//! [`events`] and [`session_ctx`] survive as the vocabulary the ~470 call sites
//! across the workspace use to describe what happened. Their emitters are now
//! inert: [`session_ctx::log_event`] and friends accept an event and drop it.
//! Keeping the shape means an event can be routed somewhere local later without
//! touching every call site again.

mod appender;
pub mod config;
pub mod context;
pub mod debug_log;
pub mod enums;
pub mod events;
pub mod hooks_log;
pub mod id;
pub mod instrumentation;
pub mod memory_log;
pub mod memory_telemetry;
pub mod prompt_timing;
pub mod sampling_log;
pub mod session_ctx;
pub mod session_metrics;
pub mod unified_log;

pub use events::TelemetryEvent;
pub use session_ctx::{
    EmitterOrigin, TelemetryCtx, emit_event, emit_event_with_origin, log_event, log_session_event,
    log_session_event_with_origin, with_session_ctx,
};
