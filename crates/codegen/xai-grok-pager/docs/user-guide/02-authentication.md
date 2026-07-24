# Authentication

Guac Build authenticates directly to the
[Meta Model API](https://ai.developer.meta.com/docs/getting-started/authentication)
with an API key. It does not use Grok browser login or an xAI subscription.

## Set an API key

Create a Meta Model API key in the
[Meta developer portal](https://ai.developer.meta.com/), then set the official
environment variable:

```sh
export MODEL_API_KEY="your-key"
guac
```

`META_API_KEY` is also accepted as a Guac-specific convenience alias.
`MODEL_API_KEY` takes precedence when both are set.

Add the export to your shell profile if you want it to persist:

```sh
# zsh
printf '%s\n' 'export MODEL_API_KEY="your-key"' >> ~/.zshrc

# bash
printf '%s\n' 'export MODEL_API_KEY="your-key"' >> ~/.bashrc
```

Restart the shell before launching `guac`.

## Configure a key per model

You can reference a different environment variable in
`~/.guac/config.toml`:

```toml
[model."muse-spark-1.1"]
env_key = "ACME_META_MODEL_API_KEY"
```

You can also set `api_key` directly, but an environment variable or secret
manager is safer because it keeps credentials out of the config file.

## Credential precedence

Guac Build resolves credentials in this order:

1. A model's explicit `api_key`.
2. The first populated variable named by the model's `env_key`.
3. `MODEL_API_KEY`, then `META_API_KEY`.

The built-in Muse Spark 1.1 model already declares
`["MODEL_API_KEY", "META_API_KEY"]`.

## Troubleshooting

Check that the variable is visible to the process:

```sh
test -n "$MODEL_API_KEY" && echo "MODEL_API_KEY is set"
```

An HTTP 401 means the key is missing, invalid, expired, or not authorized for
Meta Model API. Generate a new key in the Meta developer portal and retry.

Muse Spark 1.1 developer access is currently a public preview with regional
availability constraints.
