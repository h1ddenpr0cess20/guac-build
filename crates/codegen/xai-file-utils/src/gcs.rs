//! Shared upload utilities for session persistence and agent telemetry.
//!
//! Uploads go through the cli-chat-proxy (which resolves the destination bucket
//! from the caller's ACLs) or, when a bucket is configured, straight to S3.
//!
//! The direct-to-GCS backend upstream also carried is not built here — see
//! [`direct_upload_unsupported`].

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;

use crate::UploadMethod;
use xai_grok_auth::{AuthCredentialProvider, StaticAuthCredentialProvider};

use crate::storage_client::{Auth401AttributionCallback, StaticGrokAuth, StorageClient};

/// Threshold for switching to multipart upload (50 MB).
///
/// Files larger than this use `StorageClient::upload_multipart()` (signed URLs,
/// parts uploaded directly to cloud storage) instead of streaming through the proxy.
pub const MULTIPART_UPLOAD_THRESHOLD: u64 = 50 * 1024 * 1024;

/// Construct a `StorageClient` for proxy-mode uploads. Uses the caller-provided
/// refresh-aware credentials when present, otherwise falls back to a
/// `StaticGrokAuth` carrying the inline user / deployment keys from
/// `UploadMethod::Proxy`. The optional `http_client` lets the caller pass a
/// shell-tuned client (HTTP/2 keep-alive, conn pool tuning); when `None` we
/// fall back to `reqwest::Client::new()`.
fn build_proxy_client_with_fallback(
    proxy_base_url: &str,
    user_token: &str,
    deployment_key: Option<String>,
    credentials: Option<Arc<dyn AuthCredentialProvider>>,
    attribution: Option<Arc<dyn Auth401AttributionCallback>>,
    http_client: Option<reqwest::Client>,
) -> StorageClient {
    let provider = credentials.unwrap_or_else(|| {
        let mut creds = StaticGrokAuth::new(Some(user_token.to_owned()));
        creds.deployment_key = deployment_key;
        let bearer = creds.wire_bearer();
        Arc::new(StaticAuthCredentialProvider::new(Box::new(creds), bearer))
    });
    let http_client = http_client.unwrap_or_default();
    let mut client = StorageClient::with_provider(proxy_base_url, http_client, provider);
    if let Some(cb) = attribution {
        client = client.with_attribution(cb);
    }
    client
}

/// Implement `StorageConfig` for `TraceExportConfig`. Lives here (alongside the
/// trait + upload helpers) so callers can use the shared upload helpers without
/// a foreign-trait impl. Refresh-aware callers still get credential /
/// attribution wiring via `TraceExportConfigWithAuth` (in shell).
impl StorageConfig for crate::TraceExportConfig {
    fn bucket_url(&self) -> &str {
        // For proxy mode, bucket_url may be None (proxy determines it from ACLs).
        // Return a placeholder that won't be used.
        self.bucket_url.as_deref().unwrap_or("gs://placeholder")
    }

    fn upload_method(&self) -> &UploadMethod {
        &self.upload_method
    }
}

/// A trait for storage configuration that provides bucket URL and upload method.
/// This allows different config types (TraceExportConfig, etc.) to share upload logic.
pub trait StorageConfig {
    fn bucket_url(&self) -> &str;
    fn upload_method(&self) -> &UploadMethod;
    /// Optional refresh-aware credentials for proxy-mode uploads. When
    /// `Some(_)`, `upload_*_via_proxy` helpers construct a `StorageClient`
    /// via `StorageClient::with_provider(...)` so 401 retries can request
    /// a token refresh. Default `None` for configs that ship a static
    /// user-token only.
    fn proxy_credentials(&self) -> Option<Arc<dyn AuthCredentialProvider>> {
        None
    }
    /// Optional 401-attribution callback. When `Some(_)`, the constructed
    /// `StorageClient` also calls `with_attribution(...)` so the embedding
    /// application records auth-attribution telemetry for proxy 401s.
    fn proxy_attribution(&self) -> Option<Arc<dyn Auth401AttributionCallback>> {
        None
    }
    /// Optional HTTP client for proxy-mode uploads. `None` falls back to
    /// `reqwest::Client::new()` (used by bins/tests). Production callers
    /// should return shell's tuned `shared_upload_client()` -- HTTP/2
    /// keep-alive + aggressive connection pool eviction. The trace upload
    /// queue, feedback uploads, share uploads, and subagent metadata
    /// uploads all rely on this tuning to avoid stale-connection retries
    /// during backoff loops.
    fn proxy_http_client(&self) -> Option<reqwest::Client> {
        None
    }
}

/// Error for the direct-to-GCS upload backend, which Guac Build does not ship.
///
/// Upstream could upload traces straight to a GCS bucket using a service-account
/// key, for xAI's own trace collection. Reaching that bucket is not something
/// this fork can do, and the `gcloud-storage` client it needed pulled in its own
/// auth stack, so the backend was dropped rather than carried.
///
/// Both remaining methods are unaffected: proxy upload (the default, and the
/// only one reachable without configuring a bucket) and direct S3, which is
/// still built because `video_gen` needs the same SDK for presigned URLs.
fn direct_upload_unsupported(backend: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "{backend} direct-to-bucket upload is not available in this build. \
         Unset `trace_upload_bucket` to use proxy upload, or point it at an \
         `s3://` bucket instead."
    )
}

/// Uploads bytes to cloud storage at the specified path.
/// Returns the full storage URL on success.
/// Dispatches to the proxy backend; direct GCS/S3 are not built.
pub async fn upload_bytes<C: StorageConfig>(
    config: &C,
    object_path: &str,
    content: &[u8],
    content_type: &str,
) -> anyhow::Result<String> {
    match config.upload_method() {
        UploadMethod::Direct { .. } => Err(direct_upload_unsupported("GCS")),
        UploadMethod::Proxy {
            proxy_base_url,
            user_token,
            deployment_key,
            alpha_test_key: _,
        } => {
            // For proxy mode, bucket is determined by proxy from user ACLs
            tracing::debug!(
                proxy_base_url = %proxy_base_url,
                object_path = %object_path,
                "Uploading bytes to GCS via proxy (bucket determined by proxy from ACLs)"
            );
            upload_bytes_via_proxy(
                proxy_base_url,
                user_token,
                deployment_key.as_deref(),
                object_path,
                content,
                content_type,
                config.proxy_credentials(),
                config.proxy_attribution(),
                config.proxy_http_client(),
            )
            .await
        }
        UploadMethod::S3 {
            bucket,
            region,
            credentials_file,
            credentials_content,
            endpoint_url,
        } => {
            crate::s3::upload_bytes(
                bucket,
                object_path,
                content,
                content_type,
                region,
                credentials_content.as_deref(),
                credentials_file.as_deref(),
                endpoint_url.as_deref(),
            )
            .await
        }
    }
}

/// Like [`upload_bytes`], but in proxy mode uses a pre-signed PUT URL
/// so the data goes directly to storage instead of through the proxy.
///
/// This avoids the nginx `proxy-body-size: 4m` limit on the HTTP ingress and
/// the Cloudflare 100 MB limit, making it safe for arbitrarily large payloads
/// (e.g. session share data).
pub async fn upload_bytes_signed<C: StorageConfig>(
    config: &C,
    object_path: &str,
    content: &[u8],
    content_type: &str,
) -> anyhow::Result<String> {
    match config.upload_method() {
        UploadMethod::Direct { .. } => Err(direct_upload_unsupported("GCS")),
        UploadMethod::Proxy {
            proxy_base_url,
            user_token,
            deployment_key,
            alpha_test_key: _,
        } => {
            tracing::debug!(
                proxy_base_url = %proxy_base_url,
                object_path = %object_path,
                bytes = content.len(),
                "Uploading bytes to GCS via signed URL (bypasses proxy body limits)"
            );
            upload_bytes_via_signed_url(
                proxy_base_url,
                user_token,
                deployment_key.as_deref(),
                object_path,
                content,
                content_type,
                config.proxy_credentials(),
                config.proxy_attribution(),
                config.proxy_http_client(),
            )
            .await
        }
        UploadMethod::S3 { .. } => upload_bytes(config, object_path, content, content_type).await,
    }
}

/// Uploads a file to cloud storage by streaming from disk.
///
/// Preferred over `upload_bytes` for the background upload queue because:
/// - Never loads the full file into memory (critical for multi-GB dedup blobs)
/// - For Proxy mode with large files (>50 MB), uses signed-URL multipart upload
///   so data travels directly to storage, bypassing the proxy's body size limits
/// - For Proxy mode with small files, uses `StorageClient::upload_file()` (streaming)
pub async fn upload_file<C: StorageConfig>(
    config: &C,
    object_path: &str,
    file_path: &Path,
    content_type: &str,
) -> anyhow::Result<String> {
    match config.upload_method() {
        UploadMethod::Direct { .. } => Err(direct_upload_unsupported("GCS")),
        UploadMethod::Proxy {
            proxy_base_url,
            user_token,
            deployment_key,
            alpha_test_key: _,
        } => {
            upload_file_via_proxy(
                proxy_base_url,
                user_token,
                deployment_key.as_deref(),
                object_path,
                file_path,
                content_type,
                config.proxy_credentials(),
                config.proxy_attribution(),
                config.proxy_http_client(),
            )
            .await
        }
        UploadMethod::S3 {
            bucket,
            region,
            credentials_file,
            credentials_content,
            endpoint_url,
        } => {
            crate::s3::upload_file(
                bucket,
                object_path,
                file_path,
                content_type,
                region,
                credentials_content.as_deref(),
                credentials_file.as_deref(),
                endpoint_url.as_deref(),
            )
            .await
        }
    }
}

/// Uploads an async reader to cloud storage, dispatching to the appropriate backend.
///
/// Used for streaming compressed uploads where the reader is consumed once per attempt.
/// Callers handle retries by recreating the reader.
pub async fn upload_stream<C: StorageConfig, R>(
    config: &C,
    object_path: &str,
    reader: R,
    content_type: &str,
) -> anyhow::Result<String>
where
    R: tokio::io::AsyncRead + Send + Sync + 'static,
{
    match config.upload_method() {
        UploadMethod::Direct { .. } => Err(direct_upload_unsupported("GCS")),
        UploadMethod::Proxy {
            proxy_base_url,
            user_token,
            deployment_key,
            alpha_test_key: _,
        } => {
            let storage_client = build_proxy_client_with_fallback(
                proxy_base_url,
                user_token,
                deployment_key.as_deref().map(|s| s.to_owned()),
                config.proxy_credentials(),
                config.proxy_attribution(),
                config.proxy_http_client(),
            );
            let response = storage_client
                .upload_stream(object_path, reader, content_type)
                .await
                .with_context(|| format!("Streaming upload failed for {}", object_path))?;
            Ok(format!("gs://{}/{}", response.bucket, response.path))
        }
        UploadMethod::S3 {
            bucket,
            region,
            credentials_file,
            credentials_content,
            endpoint_url,
        } => {
            crate::s3::upload_stream(
                bucket,
                object_path,
                reader,
                content_type,
                region,
                credentials_content.as_deref(),
                credentials_file.as_deref(),
                endpoint_url.as_deref(),
            )
            .await
        }
    }
}

/// Upload a file through the cli-chat-proxy, choosing multipart vs streaming based on size.
///
/// Files > `MULTIPART_UPLOAD_THRESHOLD` use signed-URL multipart upload (parts go
/// directly to cloud storage, not through the proxy HTTP body). This avoids the proxy's request
/// body size limit and the timeout issues that cause 55% of upload failures for large
/// dedup blobs.
async fn upload_file_via_proxy(
    proxy_base_url: &str,
    user_token: &str,
    deployment_key: Option<&str>,
    object_path: &str,
    file_path: &Path,
    content_type: &str,
    credentials: Option<Arc<dyn AuthCredentialProvider>>,
    attribution: Option<Arc<dyn Auth401AttributionCallback>>,
    http_client: Option<reqwest::Client>,
) -> anyhow::Result<String> {
    use crate::storage_client::{MultipartUploadOptions, RetryConfig};

    let storage_client = build_proxy_client_with_fallback(
        proxy_base_url,
        user_token,
        deployment_key.map(|s| s.to_owned()),
        credentials,
        attribution,
        http_client,
    )
    .with_retry_config(RetryConfig::conservative());

    let file_size = tokio::fs::metadata(file_path)
        .await
        .with_context(|| format!("Failed to get file metadata: {}", file_path.display()))?
        .len();

    if file_size > MULTIPART_UPLOAD_THRESHOLD {
        // Large file: upload directly to cloud storage via signed URLs (bypasses proxy body)
        tracing::info!(
            file_size,
            threshold = MULTIPART_UPLOAD_THRESHOLD,
            upload_method = "multipart",
            path = %file_path.display(),
            "Upload queue: using multipart for large file"
        );
        let options = MultipartUploadOptions::new().with_max_concurrent(4);
        let response = storage_client
            .upload_multipart(object_path, file_path, content_type, Some(options))
            .await
            .with_context(|| format!("Multipart upload failed for {}", object_path))?;
        Ok(response.gcs_url)
    } else {
        // Small file: stream through proxy (no memory copy)
        tracing::debug!(
            file_size,
            upload_method = "streaming",
            path = %file_path.display(),
            "Upload queue: using streaming for small file"
        );
        let response = storage_client
            .upload_file(object_path, file_path, content_type)
            .await
            .with_context(|| format!("Streaming upload failed for {}", object_path))?;
        Ok(format!("gs://{}/{}", response.bucket, response.path))
    }
}

/// Uploads bytes via the cli-chat-proxy storage proxy API.
/// The bucket is determined by the proxy based on the user's ACLs.
async fn upload_bytes_via_proxy(
    proxy_base_url: &str,
    user_token: &str,
    deployment_key: Option<&str>,
    object_path: &str,
    content: &[u8],
    content_type: &str,
    credentials: Option<Arc<dyn AuthCredentialProvider>>,
    attribution: Option<Arc<dyn Auth401AttributionCallback>>,
    http_client: Option<reqwest::Client>,
) -> anyhow::Result<String> {
    use crate::storage_client::RetryConfig;

    // Conservative retry config handles storage-backend 429 errors during autoscaling.
    let storage_client = build_proxy_client_with_fallback(
        proxy_base_url,
        user_token,
        deployment_key.map(|s| s.to_owned()),
        credentials,
        attribution,
        http_client,
    )
    .with_retry_config(RetryConfig::conservative());

    let response = storage_client
        .upload(object_path, content, content_type)
        .await
        .with_context(|| {
            format!(
                "Failed to upload to storage proxy: {} (path: {})",
                proxy_base_url, object_path
            )
        })?;

    // Return the full GCS URL
    Ok(format!("gs://{}/{}", response.bucket, response.path))
}

/// Uploads bytes to cloud storage via a pre-signed PUT URL obtained from the proxy.
///
/// This completely bypasses the proxy for the data transfer, avoiding
/// nginx / Cloudflare body-size limits.  The proxy is only contacted
/// once (to generate the signed URL), after which the bytes go straight
/// to cloud storage.
///
/// Use this when the payload may exceed 4 MB (the nginx `proxy-body-size`
/// on the HTTP ingress) — e.g. session share data.
pub async fn upload_bytes_via_signed_url(
    proxy_base_url: &str,
    user_token: &str,
    deployment_key: Option<&str>,
    object_path: &str,
    content: &[u8],
    content_type: &str,
    credentials: Option<Arc<dyn AuthCredentialProvider>>,
    attribution: Option<Arc<dyn Auth401AttributionCallback>>,
    http_client: Option<reqwest::Client>,
) -> anyhow::Result<String> {
    let storage_client = build_proxy_client_with_fallback(
        proxy_base_url,
        user_token,
        deployment_key.map(|s| s.to_owned()),
        credentials,
        attribution,
        http_client,
    );

    let signed = storage_client
        .upload_bytes_signed(object_path, content, content_type)
        .await
        .with_context(|| {
            format!(
                "Failed to upload via signed URL: {} (path: {})",
                proxy_base_url, object_path
            )
        })?;

    Ok(format!("gs://{}/{}", signed.bucket, signed.path))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TraceExportConfig, UploadMethod};

    fn proxy_config() -> TraceExportConfig {
        proxy_config_with_url("https://proxy.example.com/v1".to_string())
    }

    fn proxy_config_with_url(base_url: String) -> TraceExportConfig {
        TraceExportConfig {
            bucket_url: None,
            service_account_key: None,
            upload_method: UploadMethod::Proxy {
                proxy_base_url: base_url,
                user_token: "tok".to_string(),
                deployment_key: None,
                alpha_test_key: None,
            },
            prefix_dir: None,
            gcs_prefix: None,
            absolute_paths: false,
            archive_name_override: None,
        }
    }

    fn direct_config() -> TraceExportConfig {
        TraceExportConfig {
            bucket_url: Some("gs://test-bucket".to_string()),
            service_account_key: None,
            upload_method: UploadMethod::Direct {
                service_account_key: None,
            },
            prefix_dir: None,
            gcs_prefix: None,
            absolute_paths: false,
            archive_name_override: None,
        }
    }

    #[test]
    fn multipart_threshold_is_50mb() {
        assert_eq!(
            MULTIPART_UPLOAD_THRESHOLD,
            50 * 1024 * 1024,
            "Multipart threshold must be 50 MB to match the plan and repo_changes.rs"
        );
    }

    #[tokio::test]
    async fn upload_file_proxy_missing_file_returns_error() {
        // upload_file_via_proxy checks metadata before connecting — should fail
        // fast with a descriptive error if the temp file was deleted mid-flight.
        let config = proxy_config();
        let result = upload_file(
            &config,
            "session/turn_0/test.bin",
            std::path::Path::new("/tmp/nonexistent_upload_queue_test_file"),
            "application/octet-stream",
        )
        .await;
        assert!(result.is_err(), "Should error for missing file");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("metadata") || err.contains("No such file"),
            "Error should mention file metadata: {}",
            err
        );
    }

    /// Direct-to-GCS is not built (see `direct_upload_unsupported`). It must
    /// fail with an actionable message rather than silently succeeding or
    /// reaching the network, and it must do so before touching the filesystem —
    /// note this path is nonexistent and the error is still about the backend.
    #[tokio::test]
    async fn upload_file_direct_gcs_is_unsupported() {
        let config = direct_config();
        let err = upload_file(
            &config,
            "session/turn_0/test.bin",
            std::path::Path::new("/tmp/nonexistent_upload_queue_test_file"),
            "application/octet-stream",
        )
        .await
        .expect_err("direct GCS upload must not succeed");
        let err = err.to_string();
        assert!(
            err.contains("not available in this build"),
            "error should say the backend is absent: {err}"
        );
        assert!(
            err.contains("trace_upload_bucket"),
            "error should name the setting to change: {err}"
        );
    }

    #[tokio::test]
    async fn upload_file_direct_invalid_scheme_returns_error() {
        // Verify that a non-gs:// URL is rejected before any I/O.
        let config = TraceExportConfig {
            bucket_url: Some("https://not-a-gcs-url.example.com".to_string()),
            service_account_key: None,
            upload_method: UploadMethod::Direct {
                service_account_key: None,
            },
            prefix_dir: None,
            gcs_prefix: None,
            absolute_paths: false,
            archive_name_override: None,
        };
        let result = upload_file(
            &config,
            "path/test.bin",
            std::path::Path::new("/tmp/file"),
            "application/octet-stream",
        )
        .await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not available in this build"),
            "Direct GCS is unbuilt, so the backend error precedes any URL check"
        );
    }

    /// Shared state for the dispatch test server, tracking which endpoints were hit.
    #[derive(Clone, Default)]
    struct DispatchState {
        multipart_called: std::sync::Arc<std::sync::atomic::AtomicBool>,
        storage_called: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    /// Start a minimal axum server (with proper State extractors) that records
    /// which upload routes were hit. Uses the same State extractor pattern as
    /// storage_client_tests.rs to ensure reliable flag updates in Bazel CI.
    ///
    /// Returns (addr, state) where state.multipart_called / state.storage_called
    /// are set to true when the respective route is hit.
    async fn start_dispatch_test_server() -> (std::net::SocketAddr, DispatchState) {
        use axum::{
            Router, body::Body, extract::State, http::StatusCode, response::IntoResponse,
            routing::post,
        };
        use std::sync::atomic::Ordering;
        use tokio::net::TcpListener;

        let state = DispatchState::default();

        async fn multipart_handler(
            State(s): State<DispatchState>,
            _body: Body,
        ) -> impl IntoResponse {
            s.multipart_called.store(true, Ordering::SeqCst);
            // 400 = non-retryable: client fails fast without backoff delays
            (StatusCode::BAD_REQUEST, r#"{"error":"test"}"#)
        }

        async fn storage_handler(State(s): State<DispatchState>, _body: Body) -> impl IntoResponse {
            s.storage_called.store(true, Ordering::SeqCst);
            (StatusCode::BAD_REQUEST, r#"{"error":"test"}"#)
        }

        let app = Router::new()
            .route("/v1/storage/multipart/init", post(multipart_handler))
            .route("/v1/storage", post(storage_handler))
            .with_state(state.clone());

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        // Give the server 50ms to bind and accept — more headroom for Bazel CI sandboxing.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        (addr, state)
    }

    #[tokio::test]
    async fn upload_file_via_proxy_uses_multipart_for_large_files() {
        // Large file (just over 50 MB threshold) should hit the multipart init endpoint.
        // Uses set_len() to create a sparse file — no actual disk write.
        let (addr, state) = start_dispatch_test_server().await;
        let config = proxy_config_with_url(format!("http://{}/v1", addr));

        let temp = tempfile::TempDir::new().unwrap();
        let large_file = temp.path().join("large.bin");
        let f = std::fs::File::create(&large_file).unwrap();
        f.set_len(MULTIPART_UPLOAD_THRESHOLD + 1).unwrap(); // sparse file, no actual disk write

        let _ = upload_file(
            &config,
            "session/turn_0/large.bin",
            &large_file,
            "application/octet-stream",
        )
        .await;

        assert!(
            state
                .multipart_called
                .load(std::sync::atomic::Ordering::SeqCst),
            "File > 50MB should use multipart upload"
        );
        assert!(
            !state
                .storage_called
                .load(std::sync::atomic::Ordering::SeqCst),
            "File > 50MB should NOT use the simple storage endpoint"
        );
    }

    #[tokio::test]
    async fn upload_file_via_proxy_uses_streaming_for_small_files() {
        // Small file (1 KB) should hit the simple storage endpoint, not multipart.
        let (addr, state) = start_dispatch_test_server().await;
        let config = proxy_config_with_url(format!("http://{}/v1", addr));

        let temp = tempfile::TempDir::new().unwrap();
        let small_file = temp.path().join("small.bin");
        std::fs::write(&small_file, vec![0u8; 1024]).unwrap();

        let _ = upload_file(
            &config,
            "session/turn_0/small.bin",
            &small_file,
            "application/octet-stream",
        )
        .await;

        assert!(
            !state
                .multipart_called
                .load(std::sync::atomic::Ordering::SeqCst),
            "File < 50MB should NOT use multipart upload"
        );
        assert!(
            state
                .storage_called
                .load(std::sync::atomic::Ordering::SeqCst),
            "File < 50MB should use the simple storage endpoint"
        );
    }
}
