# Contributing

Guac Build is an independent, experimental fork of
[`xai-org/grok-build`](https://github.com/xai-org/grok-build). Unlike upstream —
which does not accept external contributions — **issues and pull requests are
welcome here**, on a best-effort, hobby-project basis.

## Before you start

- The project is experimental and, per the [README](README.md), not yet
  production-ready. Expect rough edges.
- Internal `xai-*` crate names, ACP identifiers, and selected `.grok`
  compatibility paths are **intentionally retained** to keep upstream rebases
  reviewable. Don't rename them as part of an unrelated change — see
  [`docs/META_MODEL_API.md`](docs/META_MODEL_API.md) for the compatibility
  boundary.
- The root `Cargo.toml` is **generated** — prefer editing per-crate
  `Cargo.toml` files.

## Development

```sh
cargo check -p <crate>        # target specific crates; full-workspace builds are slow
cargo test -p <crate>         # per-crate tests
cargo clippy -p <crate>       # lint config: clippy.toml at the repo root
cargo fmt --all               # rustfmt.toml at the repo root
cargo build -p xai-grok-pager-bin --bin guac   # build the binary
```

CI runs `cargo fmt --all --check`, builds the `guac` binary, and runs the
built-binary end-to-end smoke tests against the in-tree mock inference server.
Please make sure `cargo fmt --all --check` passes before opening a PR.

## Pull requests

- Keep changes focused; describe what and why (the PR template will prompt you).
- Include tests for behavior changes where practical.
- Note anything that affects the provider, default model, auth, or the
  upstream-compatibility boundary.

## Security reports

Please report security issues through the process described in
[`SECURITY.md`](SECURITY.md). Do not open a public issue for vulnerabilities.

## Licensing

First-party code is licensed under the Apache License, Version 2.0 (see
[`LICENSE`](LICENSE)). By submitting a contribution, you agree that it is
licensed under the same terms.
