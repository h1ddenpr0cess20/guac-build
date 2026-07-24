# Meta Model API integration

Guac Build defaults to Meta's `muse-spark-1.1` model through the
OpenAI-compatible Responses API:

```text
https://api.meta.ai/v1
```

Set the official API-key environment variable before starting Guac:

```sh
export MODEL_API_KEY="..."
guac
```

`META_API_KEY` is accepted as a convenience alias. The upstream
`XAI_API_KEY` and `GROK_CODE_XAI_API_KEY` names are accepted only as migration
fallbacks and are not advertised.

## Tool mapping

| Capability | Guac implementation |
|---|---|
| Web search | Meta Responses API hosted tool: `{"type":"web_search"}` |
| Code execution | Guac's local, policy-controlled terminal and sandbox tools |
| File operations | Guac's local read, search, edit, patch, and directory tools |
| MCP tools | Guac's existing MCP client and per-project server configuration |
| Skills and subagents | Guac's existing local orchestration layer |
| Image, video, and PDF input | Passed through the model's multimodal request path |
| X search | Disabled for built-in agents because it is xAI-specific |

Meta documents web search as an API-hosted tool. Code execution is instead
composed on the agent side, so Guac keeps the mature local execution and
sandbox implementation inherited from Grok Build.

## Disabled upstream services

Guac does not use xAI browser login, hosted setup, conversation sharing, or
the upstream auto-update service. Authentication is API-key only by default.
Feedback and telemetry submission are disabled by default.

## Compatibility boundary

The executable, default storage home, model, provider endpoint, user-facing
copy, and default agent behavior are Guac-branded. Internal `xai-*` crate
names, ACP extension identifiers, and selected `.grok` project compatibility
paths remain intact to reduce client breakage and keep upstream rebases
reviewable.

## References

- [Meta Model API overview](https://ai.developer.meta.com/docs/getting-started/overview)
- [Muse Spark models](https://ai.developer.meta.com/docs/getting-started/models/)
- [Authentication](https://ai.developer.meta.com/docs/getting-started/authentication)
- [Search grounding](https://ai.developer.meta.com/docs/getting-started/cookbook/search-grounding/)
