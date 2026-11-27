#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::io::ErrorKind;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use uuid::Uuid;
use zervram_hal::{
    connect_physical_backend_with_options, AcceleratorDevice, BackendKind, DeviceDescriptor,
    DeviceState, HalError, PhysicalBackendOptions,
};
use zervram_scrub::{
    default_chain_file_path, default_quarantine_file_path, hash_tenant_id, AppendOnlyChainStore,
    AttestationBundle, AttestationSigner, CertificateStore, Ed25519Signer, QuarantineStore,
    ScrubConfig, ScrubError, ScrubRequest, VramScrubber, VramZeroEngine,
};

const MAX_HEADER_BYTES: usize = 16 * 1024;
const MAX_BODY_BYTES: usize = 64 * 1024;

#[derive(Debug, Error)]
enum DaemonError {
    #[error("HAL error: {0}")]
    Hal(#[from] HalError),
    #[error("scrub error: {0}")]
    Scrub(#[from] ScrubError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Config(String),
}

#[derive(Debug, Parser)]
#[command(
    name = "zervramd",
    version,
    about = "Persistent warm GPU VRAM sanitization daemon (localhost HTTP/JSON RPC)"
)]
struct Cli {
    /// Localhost TCP address to bind (default: 127.0.0.1:9740).
    #[arg(long, default_value = "127.0.0.1:9740", env = "ZERVRAM_DAEMON_LISTEN")]
    listen: String,

    /// Explicitly authorize binding a non-loopback interface (requires bearer token).
    #[arg(long, default_value_t = false)]
    allow_non_loopback: bool,

    /// Explicitly authorize software mock-gpu backend when no physical CUDA/ROCm GPU is present.
    #[arg(long, default_value_t = false)]
    allow_mock: bool,

    /// Bypass fail-closed NVML active-process safety gate.
    #[arg(long, default_value_t = false)]
    force: bool,

    /// Percentage of free physical VRAM to sweep (1..=100, default: 99).
    #[arg(long)]
    sweep_ratio_pct: Option<u8>,

    /// State directory for `chain.jsonl` and `quarantine.json` when individual paths are omitted.
    #[arg(long)]
    state_dir: Option<PathBuf>,

    /// Persistent PKCS#8 Ed25519 signing key file.
    #[arg(long, default_value = "node.key")]
    key_file: PathBuf,

    /// Persistent append-only JSONL hash-chain file.
    #[arg(long)]
    chain_file: Option<PathBuf>,

    /// Persistent quarantine registry JSON file.
    #[arg(long)]
    quarantine_file: Option<PathBuf>,

    /// Optional file containing bearer token for local RPC authentication.
    #[arg(long)]
    token_file: Option<PathBuf>,

    /// Optional bearer token string (or `ZERVRAM_DAEMON_TOKEN` env var).
    #[arg(long, env = "ZERVRAM_DAEMON_TOKEN", hide_env_values = true)]
    token: Option<String>,

    /// Optional file where the bound `http://<addr>` URL is atomically written once listening.
    #[arg(long)]
    addr_file: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    version: String,
    backend_kind: BackendKind,
    warm_device_count: usize,
    head_sha256: String,
    ed25519_public_key_hex: String,
    any_quarantined: bool,
    quarantined_device_uuids: Vec<Uuid>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScrubRpcRequest {
    #[serde(default)]
    device: usize,
    #[serde(default)]
    allow_mock: bool,
    #[serde(default)]
    challenge_nonce: Option<String>,
    #[serde(default)]
    tenant_id: Option<String>,
    #[serde(default)]
    job_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    code: &'static str,
    error: String,
    quarantined: bool,
}

struct DaemonState {
    hal: Arc<dyn AcceleratorDevice>,
    engine: VramZeroEngine,
    chain_store: Arc<AppendOnlyChainStore>,
    quarantine_store: Arc<QuarantineStore>,
    pubkey_hex: String,
    devices: Vec<DeviceDescriptor>,
    device_locks: BTreeMap<usize, Arc<Mutex<()>>>,
    bearer_token: Option<String>,
}

struct ParsedHttpRequest {
    method: String,
    path: String,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
}

fn select_backend(
    allow_mock: bool,
    options: PhysicalBackendOptions,
) -> Result<Arc<dyn AcceleratorDevice>, DaemonError> {
    match connect_physical_backend_with_options(options) {
        Ok(physical) => Ok(physical),
        Err(phys_err) => {
            if !allow_mock {
                return Err(DaemonError::Hal(phys_err));
            }
            #[cfg(feature = "mock-gpu")]
            {
                Ok(Arc::new(zervram_hal::MockAcceleratorBackend::new(
                    zervram_hal::MockClusterConfig::default(),
                )))
            }
            #[cfg(not(feature = "mock-gpu"))]
            {
                Err(DaemonError::Config(format!(
                    "--allow-mock requested, but binary was compiled without `mock-gpu` ({phys_err})"
                )))
            }
        }
    }
}

fn load_bearer_token(
    token_file: Option<&Path>,
    cli_token: Option<&str>,
) -> Result<Option<String>, DaemonError> {
    if let Some(path) = token_file {
        let raw = std::fs::read_to_string(path)?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(DaemonError::Config(format!(
                "--token-file '{}' is empty",
                path.display()
            )));
        }
        return Ok(Some(trimmed.to_string()));
    }
    if let Some(tok) = cli_token {
        let trimmed = tok.trim();
        if !trimmed.is_empty() {
            return Ok(Some(trimmed.to_string()));
        }
    }
    Ok(None)
}

fn constant_time_token_eq(expected: &str, provided: &str) -> bool {
    let a = Sha256::digest(expected.as_bytes());
    let b = Sha256::digest(provided.as_bytes());
    let mut diff: u8 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0 && expected.len() == provided.len()
}

fn check_bearer_auth(expected: Option<&str>, headers: &BTreeMap<String, String>) -> bool {
    let Some(expected_tok) = expected else {
        return true;
    };
    let Some(auth) = headers.get("authorization") else {
        return false;
    };
    let mut parts = auth.split_whitespace();
    let Some(scheme) = parts.next() else {
        return false;
    };
    let Some(provided) = parts.next() else {
        return false;
    };
    if parts.next().is_some() || !scheme.eq_ignore_ascii_case("bearer") {
        return false;
    }
    constant_time_token_eq(expected_tok, provided)
}

async fn read_some(stream: &TcpStream, buf: &mut [u8]) -> std::io::Result<usize> {
    loop {
        stream.readable().await?;
        match stream.try_read(buf) {
            Ok(n) => return Ok(n),
            Err(err) if err.kind() == ErrorKind::WouldBlock => continue,
            Err(err) if err.kind() == ErrorKind::Interrupted => continue,
            Err(err) => return Err(err),
        }
    }
}

async fn write_all(stream: &TcpStream, mut data: &[u8]) -> std::io::Result<()> {
    while !data.is_empty() {
        stream.writable().await?;
        match stream.try_write(data) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    ErrorKind::WriteZero,
                    "short TCP write",
                ));
            }
            Ok(n) => data = data.get(n..).unwrap_or(&[]),
            Err(err) if err.kind() == ErrorKind::WouldBlock => continue,
            Err(err) if err.kind() == ErrorKind::Interrupted => continue,
            Err(err) => return Err(err),
        }
    }
    Ok(())
}

async fn read_http_request(stream: &TcpStream) -> Result<ParsedHttpRequest, (u16, &'static str, String)> {
    let mut raw: Vec<u8> = Vec::with_capacity(1024);
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        if let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos;
        }
        if raw.len() > MAX_HEADER_BYTES {
            return Err((413, "Payload Too Large", "HTTP headers exceeded 16 KiB".to_string()));
        }
        let n = read_some(stream, &mut chunk)
            .await
            .map_err(|e| (400, "Bad Request", format!("read error: {e}")))?;
        if n == 0 {
            return Err((400, "Bad Request", "unexpected EOF reading HTTP headers".to_string()));
        }
        if let Some(slice) = chunk.get(..n) {
            raw.extend_from_slice(slice);
        }
    };

    let header_bytes = raw
        .get(..header_end)
        .ok_or_else(|| (400, "Bad Request", "malformed HTTP headers".to_string()))?;
    let header_text = std::str::from_utf8(header_bytes)
        .map_err(|_| (400, "Bad Request", "non-UTF8 HTTP headers".to_string()))?;

    let mut lines = header_text.split("\r\n");
    let req_line = lines
        .next()
        .ok_or_else(|| (400, "Bad Request", "missing HTTP request line".to_string()))?;
    let mut req_parts = req_line.split_whitespace();
    let method = req_parts
        .next()
        .ok_or_else(|| (400, "Bad Request", "missing HTTP method".to_string()))?
        .to_string();
    let target = req_parts
        .next()
        .ok_or_else(|| (400, "Bad Request", "missing HTTP path".to_string()))?;
    let version = req_parts
        .next()
        .ok_or_else(|| (400, "Bad Request", "missing HTTP version".to_string()))?;
    if !version.starts_with("HTTP/1.") {
        return Err((400, "Bad Request", format!("unsupported HTTP version '{version}'")));
    }
    let path = target.split_once('?').map_or(target, |(p, _)| p).to_string();

    let mut headers = BTreeMap::new();
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }

    let content_len = match headers.get("content-length") {
        Some(v) => v
            .parse::<usize>()
            .map_err(|_| (400, "Bad Request", "invalid Content-Length header".to_string()))?,
        None => 0,
    };
    if content_len > MAX_BODY_BYTES {
        return Err((413, "Payload Too Large", "request body exceeded 64 KiB".to_string()));
    }

    let body_start = header_end.saturating_add(4);
    let mut body = raw.get(body_start..).unwrap_or(&[]).to_vec();
    while body.len() < content_len {
        let n = read_some(stream, &mut chunk)
            .await
            .map_err(|e| (400, "Bad Request", format!("body read error: {e}")))?;
        if n == 0 {
            return Err((400, "Bad Request", "unexpected EOF reading HTTP body".to_string()));
        }
        if let Some(slice) = chunk.get(..n) {
            body.extend_from_slice(slice);
        }
    }
    body.truncate(content_len);

    Ok(ParsedHttpRequest {
        method,
        path,
        headers,
        body,
    })
}

async fn send_json<T: Serialize>(stream: &TcpStream, status: u16, reason: &str, payload: &T) {
    let body = serde_json::to_vec(payload).unwrap_or_else(|_| b"{}".to_vec());
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = write_all(stream, head.as_bytes()).await;
    let _ = write_all(stream, &body).await;
}

async fn handle_health(stream: &TcpStream, state: &DaemonState) {
    let head_sha256 = match state.chain_store.latest_hash().await {
        Ok(h) => h,
        Err(err) => {
            send_json(
                stream,
                500,
                "Internal Server Error",
                &ErrorResponse {
                    code: "storage_error",
                    error: err.to_string(),
                    quarantined: false,
                },
            )
            .await;
            return;
        }
    };

    let mut quarantined_uuids = Vec::new();
    for dev in &state.devices {
        if let Ok(Some(_)) = state.quarantine_store.is_quarantined(dev.device_uuid) {
            quarantined_uuids.push(dev.device_uuid);
        }
    }

    send_json(
        stream,
        200,
        "OK",
        &HealthResponse {
            version: env!("CARGO_PKG_VERSION").to_string(),
            backend_kind: state.hal.backend_kind(),
            warm_device_count: state.devices.len(),
            head_sha256,
            ed25519_public_key_hex: state.pubkey_hex.clone(),
            any_quarantined: !quarantined_uuids.is_empty(),
            quarantined_device_uuids: quarantined_uuids,
        },
    )
    .await;
}

async fn handle_scrub(stream: &TcpStream, state: &DaemonState, body: &[u8]) {
    let req: ScrubRpcRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(err) => {
            send_json(
                stream,
                400,
                "Bad Request",
                &ErrorResponse {
                    code: "invalid_json",
                    error: format!("invalid /v1/scrub JSON payload: {err}"),
                    quarantined: false,
                },
            )
            .await;
            return;
        }
    };

    if state.hal.backend_kind() == BackendKind::MockCpu && !req.allow_mock {
        send_json(
            stream,
            403,
            "Forbidden",
            &ErrorResponse {
                code: "mock_not_allowed",
                error: "daemon is running software mock-gpu backend; request must set allow_mock=true"
                    .to_string(),
                quarantined: false,
            },
        )
        .await;
        return;
    }

    let (Some(target), Some(dev_lock)) = (state.devices.get(req.device), state.device_locks.get(&req.device)) else {
        send_json(
            stream,
            404,
            "Not Found",
            &ErrorResponse {
                code: "invalid_device",
                error: format!(
                    "requested device {} out of range ({} device(s))",
                    req.device,
                    state.devices.len()
                ),
                quarantined: false,
            },
        )
        .await;
        return;
    };

    // Serialize concurrent scrubs per device ordinal.
    let _dev_guard = dev_lock.lock().await;

    match state.quarantine_store.is_quarantined(target.device_uuid) {
        Ok(Some(rec)) => {
            send_json(
                stream,
                423,
                "Locked",
                &ErrorResponse {
                    code: "quarantined",
                    error: format!("device {} is quarantined: {}", target.device_uuid, rec.reason),
                    quarantined: true,
                },
            )
            .await;
            return;
        }
        Ok(None) => {}
        Err(err) => {
            send_json(
                stream,
                500,
                "Internal Server Error",
                &ErrorResponse {
                    code: "storage_error",
                    error: err.to_string(),
                    quarantined: false,
                },
            )
            .await;
            return;
        }
    }

    let tenant_id = req
        .tenant_id
        .unwrap_or_else(|| "epilog-prev-tenant".to_string());
    let job_id = req.job_id.unwrap_or_else(|| "epilog-job".to_string());

    if state.hal.backend_kind() == BackendKind::MockCpu {
        match state.hal.get_device(target.device_uuid).await {
            Ok(desc) => match desc.state {
                DeviceState::Quarantined { reason, .. } => {
                    send_json(
                        stream,
                        423,
                        "Locked",
                        &ErrorResponse {
                            code: "quarantined",
                            error: format!("device {} is quarantined: {reason}", target.device_uuid),
                            quarantined: true,
                        },
                    )
                    .await;
                    return;
                }
                DeviceState::Available => {
                    if let Err(err) = state
                        .hal
                        .allocate_slice(
                            target.device_uuid,
                            hash_tenant_id(&tenant_id),
                            job_id.clone(),
                        )
                        .await
                    {
                        send_json(
                            stream,
                            500,
                            "Internal Server Error",
                            &ErrorResponse {
                                code: "hal_error",
                                error: err.to_string(),
                                quarantined: false,
                            },
                        )
                        .await;
                        return;
                    }
                }
                _ => {}
            },
            Err(err) => {
                send_json(
                    stream,
                    500,
                    "Internal Server Error",
                    &ErrorResponse {
                        code: "hal_error",
                        error: err.to_string(),
                        quarantined: false,
                    },
                )
                .await;
                return;
            }
        }
    }

    let cert_res = state
        .engine
        .sanitize_on_job_exit(ScrubRequest {
            device_uuid: target.device_uuid,
            job_id,
            prev_tenant_id: tenant_id,
            explicit_range: None,
            challenge_nonce: req.challenge_nonce,
        })
        .await;

    match cert_res {
        Ok(cert) => match cert.certificate_sha256_hex() {
            Ok(cert_sha) => {
                let bundle = AttestationBundle {
                    ed25519_public_key_hex: state.pubkey_hex.clone(),
                    certificate_sha256: cert_sha,
                    certificate: cert,
                };
                send_json(stream, 200, "OK", &bundle).await;
            }
            Err(err) => {
                send_json(
                    stream,
                    500,
                    "Internal Server Error",
                    &ErrorResponse {
                        code: "hash_error",
                        error: err.to_string(),
                        quarantined: false,
                    },
                )
                .await;
            }
        },
        Err(
            err @ (ScrubError::DeviceQuarantined { .. }
            | ScrubError::ResidualDataDetected { .. }
            | ScrubError::Hal(HalError::HardwareFaultQuarantine { .. })),
        ) => {
            send_json(
                stream,
                423,
                "Locked",
                &ErrorResponse {
                    code: "quarantined",
                    error: err.to_string(),
                    quarantined: true,
                },
            )
            .await;
        }
        Err(err @ ScrubError::Hal(HalError::ActiveProcessesDetected { .. })) => {
            send_json(
                stream,
                409,
                "Conflict",
                &ErrorResponse {
                    code: "active_processes",
                    error: err.to_string(),
                    quarantined: false,
                },
            )
            .await;
        }
        Err(err) => {
            send_json(
                stream,
                500,
                "Internal Server Error",
                &ErrorResponse {
                    code: "scrub_failed",
                    error: err.to_string(),
                    quarantined: false,
                },
            )
            .await;
        }
    }
}

async fn handle_connection(stream: TcpStream, state: Arc<DaemonState>) {
    let req = match tokio::time::timeout(Duration::from_secs(10), read_http_request(&stream)).await
    {
        Ok(Ok(r)) => r,
        Ok(Err((status, reason, msg))) => {
            send_json(
                &stream,
                status,
                reason,
                &ErrorResponse {
                    code: "bad_request",
                    error: msg,
                    quarantined: false,
                },
            )
            .await;
            return;
        }
        Err(_) => {
            send_json(
                &stream,
                408,
                "Request Timeout",
                &ErrorResponse {
                    code: "timeout",
                    error: "timed out reading HTTP request".to_string(),
                    quarantined: false,
                },
            )
            .await;
            return;
        }
    };

    if !check_bearer_auth(state.bearer_token.as_deref(), &req.headers) {
        send_json(
            &stream,
            401,
            "Unauthorized",
            &ErrorResponse {
                code: "unauthorized",
                error: "missing or invalid Authorization: Bearer token".to_string(),
                quarantined: false,
            },
        )
        .await;
        return;
    }

    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/v1/health") => handle_health(&stream, &state).await,
        ("POST", "/v1/scrub") => handle_scrub(&stream, &state, &req.body).await,
        (_, "/v1/health" | "/v1/scrub") => {
            send_json(
                &stream,
                405,
                "Method Not Allowed",
                &ErrorResponse {
                    code: "method_not_allowed",
                    error: format!("method {} not allowed on {}", req.method, req.path),
                    quarantined: false,
                },
            )
            .await;
        }
        _ => {
            send_json(
                &stream,
                404,
                "Not Found",
                &ErrorResponse {
                    code: "not_found",
                    error: format!("unknown route {}", req.path),
                    quarantined: false,
                },
            )
            .await;
        }
    }
}

fn write_addr_file_atomically(path: &Path, url: &str) -> Result<(), DaemonError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut tmp_os = path.as_os_str().to_os_string();
    tmp_os.push(format!(".tmp.{}", std::process::id()));
    let tmp_path = PathBuf::from(tmp_os);
    std::fs::write(&tmp_path, url.as_bytes())?;
    #[cfg(target_os = "windows")]
    if path.exists() {
        let _ = std::fs::remove_file(path);
    }
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

async fn run(cli: Cli) -> Result<(), DaemonError> {
    let listen_addr: SocketAddr = cli
        .listen
        .parse()
        .map_err(|e| DaemonError::Config(format!("invalid --listen '{}': {e}", cli.listen)))?;
    let bearer_token = load_bearer_token(cli.token_file.as_deref(), cli.token.as_deref())?;

    if !listen_addr.ip().is_loopback() {
        if !cli.allow_non_loopback {
            return Err(DaemonError::Config(format!(
                "refusing to bind non-loopback address {listen_addr} without --allow-non-loopback"
            )));
        }
        if bearer_token.is_none() {
            return Err(DaemonError::Config(
                "--allow-non-loopback requires --token-file or ZERVRAM_DAEMON_TOKEN".to_string(),
            ));
        }
    }

    let (signer_obj, created_key) = Ed25519Signer::load_or_generate_key_file(&cli.key_file)?;
    let pubkey_hex = signer_obj.verifying_key_hex();
    if created_key {
        eprintln!(
            "zervramd: generated persistent Ed25519 key at {} (pubkey={})",
            cli.key_file.display(),
            pubkey_hex
        );
    }

    let chain_path = match (cli.chain_file, cli.state_dir.as_ref()) {
        (Some(p), _) => p,
        (None, Some(dir)) => dir.join("chain.jsonl"),
        (None, None) => default_chain_file_path(),
    };
    let quarantine_path = match (cli.quarantine_file, cli.state_dir.as_ref()) {
        (Some(p), _) => p,
        (None, Some(dir)) => dir.join("quarantine.json"),
        (None, None) => default_quarantine_file_path(Some(&chain_path)),
    };

    let phys_opts = PhysicalBackendOptions {
        force_ignore_active_processes: cli.force,
        sweep_free_vram_pct: cli.sweep_ratio_pct,
    };
    let hal = select_backend(cli.allow_mock, phys_opts)?;
    let devices = hal.discover_devices().await?;
    let mut device_locks = BTreeMap::new();
    for idx in 0..devices.len() {
        device_locks.insert(idx, Arc::new(Mutex::new(())));
    }

    let chain_store = Arc::new(AppendOnlyChainStore::open(&chain_path)?);
    let quarantine_store = Arc::new(QuarantineStore::open(&quarantine_path));
    let signer: Arc<dyn AttestationSigner> = Arc::new(signer_obj);
    let store: Arc<dyn CertificateStore> = Arc::clone(&chain_store) as Arc<dyn CertificateStore>;
    let config = match hal.backend_kind() {
        BackendKind::MockCpu => ScrubConfig::default(),
        _ => ScrubConfig::production(600_000),
    };
    let engine = VramZeroEngine::new(Arc::clone(&hal), signer, store, config)
        .with_quarantine_store(Arc::clone(&quarantine_store));

    let listener = TcpListener::bind(listen_addr).await?;
    let bound_addr = listener.local_addr()?;
    let bound_url = format!("http://{bound_addr}");
    if let Some(ref addr_file) = cli.addr_file {
        write_addr_file_atomically(addr_file, &bound_url)?;
    }

    let state = Arc::new(DaemonState {
        hal: Arc::clone(&hal),
        engine,
        chain_store,
        quarantine_store,
        pubkey_hex: pubkey_hex.clone(),
        devices,
        device_locks,
        bearer_token,
    });

    eprintln!(
        "zervramd v{} listening on {} (backend={:?}, devices={}, pubkey={})",
        env!("CARGO_PKG_VERSION"),
        bound_url,
        state.hal.backend_kind(),
        state.devices.len(),
        pubkey_hex
    );

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                eprintln!("zervramd: shutting down cleanly");
                break;
            }
            accepted = listener.accept() => {
                match accepted {
                    Ok((stream, _peer)) => {
                        let _ = stream.set_nodelay(true);
                        let st = Arc::clone(&state);
                        tokio::spawn(async move {
                            handle_connection(stream, st).await;
                        });
                    }
                    Err(err) => {
                        eprintln!("zervramd: accept error: {err}");
                    }
                }
            }
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}