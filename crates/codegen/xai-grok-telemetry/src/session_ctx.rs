//! Ambient session context, and the (now inert) event emitters.
//!
//! `session_id` and `turn_number` come from the task-local [`TelemetryCtx`]
//! active for the duration of a session, and still feed the local tracing span
//! the debug-log firehose routes by.
//!
//! The emitters below used to fan out to two network sinks: xAI's product
//! events endpoint plus Mixpanel, and an OpenTelemetry exporter. Both are gone.
//! The functions remain so the ~470 call sites that describe session activity
//! keep compiling and keep documenting what happens where -- they accept a
//! typed event and drop it.

use std::sync::Arc;

use serde::Serialize;

use crate::events::TelemetryEvent;

/// Ambient session context for telemetry. Snapshotted synchronously by
/// `log_event` at call time to avoid racing with turn increments.
#[derive(Clone)]
pub struct TelemetryCtx {
    pub session_id: String,
    pub prompt_index: Arc<tokio::sync::Mutex<usize>>,
    /// Per-prompt correlation UUID. Set at turn start where `prompt_index`
    /// increments; `None` outside a prompt.
    pub prompt_id: Arc<parking_lot::Mutex<Option<String>>>,
}

impl TelemetryCtx {
    pub fn new(session_id: String, prompt_index: Arc<tokio::sync::Mutex<usize>>) -> Self {
        Self {
            session_id,
            prompt_index,
            prompt_id: Arc::new(parking_lot::Mutex::new(None)),
        }
    }
}

/// Rotate the per-prompt correlation UUID at turn start (where
/// `prompt_index` increments). No-op outside a session ctx scope.
pub fn begin_prompt_id() {
    let _ = TELEMETRY_CTX.try_with(|c| {
        *c.prompt_id.lock() = Some(uuid::Uuid::new_v4().to_string());
    });
}

tokio::task_local! {
    static TELEMETRY_CTX: Arc<TelemetryCtx>;
}

/// The `session_id` field name the debug-log firehose router keys on:
/// `debug_log::SessionIdVisitor` stashes a `SessionId` extension on any span
/// carrying this field — the span *name* is not load-bearing for routing. Shared
/// so the `info_span!` here and the router in `debug_log` can't silently drift; a
/// rename trips `session_span_exposes_router_field` below.
pub(crate) const SESSION_ID_FIELD: &str = "session_id";

/// Build the per-session tracing span the firehose router routes by. The field
/// name MUST be the literal `session_id` (tracing field names can't come from a
/// const); the test below pins it against [`SESSION_ID_FIELD`].
fn session_span(session_id: &str) -> tracing::Span {
    tracing::info_span!("session", session_id = %session_id)
}

/// Run `fut` with telemetry context active. Also sets a `tracing` span.
pub async fn with_session_ctx<F: std::future::Future>(ctx: TelemetryCtx, fut: F) -> F::Output {
    use tracing::Instrument;
    let span = session_span(&ctx.session_id);
    TELEMETRY_CTX
        .scope(Arc::new(ctx), fut.instrument(span))
        .await
}

/// Product surface that emitted an event. Selects the event-name prefix so
/// shell and workspace events stay distinguishable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::EnumCount)]
pub enum EmitterOrigin {
    /// `xai-grok-shell` (and the pager/TUI that emit through it).
    Shell,
    /// `xai-grok-workspace` (remote sampler / workspace server).
    Workspace,
}

impl EmitterOrigin {
    /// Every emitter origin. Completeness is compiler-enforced by the length
    /// assertion below, so a newly added variant omitted here fails to compile.
    pub const ALL: [EmitterOrigin; 2] = [EmitterOrigin::Shell, EmitterOrigin::Workspace];

    /// Event-name prefix for this origin.
    pub fn event_prefix(self) -> &'static str {
        match self {
            EmitterOrigin::Shell => "grok-shell-",
            EmitterOrigin::Workspace => "grok-workspace-",
        }
    }
}

/// Compile-time completeness guard for [`EmitterOrigin::ALL`]: adding a variant
/// without listing it in `ALL` makes `ALL.len()` diverge from the
/// `strum::EnumCount`-derived variant count and fails this assertion.
const _: () = assert!(EmitterOrigin::ALL.len() == <EmitterOrigin as strum::EnumCount>::COUNT);

/// Record a typed event. Inert: the product-events/Mixpanel funnel and the
/// OpenTelemetry exporter this used to feed were both removed, so the event is
/// accepted and dropped. Call sites are kept because they document, in one
/// place per behaviour, what the session just did.
pub fn log_event<T: TelemetryEvent>(_data: T) {}

/// Inert. Previously split one event across an internal and an external sink
/// under separate gates; both sinks are gone, so the gate is meaningless.
pub fn log_event_dual<T: TelemetryEvent>(_internal_enabled: bool, _data: T) {}

/// Inert. See [`log_event`].
pub fn log_session_event<T: TelemetryEvent>(_data: T) {}

/// Inert. See [`log_event`].
pub fn log_session_event_with_origin<T: TelemetryEvent>(_origin: EmitterOrigin, _data: T) {}

/// Inert. See [`log_event`].
pub fn emit_event<T: Serialize + Send + 'static>(_event_suffix: impl Into<String>, _data: T) {}

/// Inert. See [`log_event`].
pub fn emit_event_with_origin<T: Serialize + Send + 'static>(
    _origin: EmitterOrigin,
    _event_suffix: impl Into<String>,
    _data: T,
) {
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The debug-log firehose router (`debug_log`) finds the session span by its
    /// `session_id` field (not by name). That field name is a literal in
    /// `session_span` (tracing field names can't be a const), so pin it against the
    /// shared const here — a rename of either breaks this test instead of silently
    /// degrading routing to the per-pid fallback.
    #[test]
    fn session_span_exposes_router_field() {
        // A bare registry enables every callsite, so the span has live metadata.
        let subscriber = tracing_subscriber::registry();
        tracing::subscriber::with_default(subscriber, || {
            let span = session_span("test-id");
            let meta = span
                .metadata()
                .expect("session span must have metadata under an enabling subscriber");
            assert!(
                meta.fields().field(SESSION_ID_FIELD).is_some(),
                "session span must expose `{SESSION_ID_FIELD}` for debug-log routing",
            );
        });
    }

    /// Event-name prefixes must not drift.
    #[test]
    fn event_prefix_is_stable_per_origin() {
        assert_eq!(EmitterOrigin::Shell.event_prefix(), "grok-shell-");
        assert_eq!(EmitterOrigin::Workspace.event_prefix(), "grok-workspace-");
    }

    /// The `Shell` reroute must reproduce the historical
    /// `format!("grok-shell-{suffix}")` event name byte-for-byte.
    #[test]
    fn shell_origin_event_name_matches_legacy_format() {
        let suffix = "trace_upload_attempted";
        let rerouted = format!("{}{}", EmitterOrigin::Shell.event_prefix(), suffix);
        let legacy = format!("grok-shell-{suffix}");
        assert_eq!(rerouted, legacy);
    }

    #[test]
    fn workspace_origin_event_name_uses_workspace_prefix() {
        let name = format!("{}turn", EmitterOrigin::Workspace.event_prefix());
        assert_eq!(name, "grok-workspace-turn");
    }

    /// Length completeness is compiler-enforced by the `const _` assertion in
    /// this module (via `strum::EnumCount`); this test additionally pins that
    /// the known variants are present and that every origin yields a distinct,
    /// non-empty prefix (which `EnumCount` alone does not guarantee).
    #[test]
    fn all_covers_every_origin_with_distinct_nonempty_prefixes() {
        assert!(EmitterOrigin::ALL.contains(&EmitterOrigin::Shell));
        assert!(EmitterOrigin::ALL.contains(&EmitterOrigin::Workspace));
        assert_eq!(
            EmitterOrigin::ALL.len(),
            <EmitterOrigin as strum::EnumCount>::COUNT,
            "ALL must list every EmitterOrigin variant",
        );

        let mut prefixes: Vec<&str> = EmitterOrigin::ALL
            .iter()
            .map(|o| o.event_prefix())
            .collect();
        assert!(
            prefixes.iter().all(|p| !p.is_empty()),
            "every origin must have a non-empty prefix",
        );
        let total = prefixes.len();
        prefixes.sort_unstable();
        prefixes.dedup();
        assert_eq!(
            prefixes.len(),
            total,
            "every origin must yield a distinct prefix",
        );
    }
}
