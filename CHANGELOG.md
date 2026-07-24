# Changelog

Notable changes in Guac Build relative to its upstream, `xai-org/grok-build`.
This project is experimental; versions track the inherited crate version
(`guac --version`) and are not yet formally released.

## Unreleased

### Fork (Guac Build)

- **Provider/model:** default to Meta's `muse-spark-1.1` over the
  OpenAI-compatible Meta Model API (`https://api.meta.ai/v1`); enable backend
  search grounding; remove xAI's hosted `x_search` from the built-in agents.
- **Auth:** API-key only by default via `MODEL_API_KEY` (with `META_API_KEY`
  alias); upstream `XAI_API_KEY` / `GROK_CODE_XAI_API_KEY` accepted as migration
  fallbacks. Disabled xAI browser login, hosted setup, session sharing, and the
  auto-updater. Feedback/telemetry default off.
- **Branding:** binary renamed to `guac`; config home is `~/.guac`
  (`GUAC_HOME`, with `GROK_HOME` compatibility). Internal `xai-*` crate names
  retained to keep upstream rebases reviewable.

### Fixed

- `guac leader kill` skipped every live leader (and deleted its lock/socket as
  "stale") because the process matcher only recognized the old `grok` binary
  name. It now matches `guac` (with `grok` kept for compatibility); added a
  regression test for the matcher.
- Test/PTY harnesses (`xai-grok-test-support`, `xai-grok-pager-pty-harness`,
  `doctor_early_dispatch`) referenced the old `xai-grok-pager` binary name and
  `CARGO_BIN_EXE_xai-grok-pager`, so the headless/PTY e2e suites could not
  locate the built binary. Repointed to `guac` / `CARGO_BIN_EXE_guac`.
- Fixed rustfmt violations introduced by the rebrand commit.

### Added

- CI (`cargo fmt --all --check`, build `guac`, built-binary e2e smoke tests),
  a tag-triggered release workflow, Dependabot, and issue/PR templates.
