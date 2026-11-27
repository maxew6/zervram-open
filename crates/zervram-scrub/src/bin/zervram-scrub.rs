#![forbid(unsafe_code)]

use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, Subcommand};
use ed25519_dalek::VerifyingKey;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::net::TcpStream;
use zervram_hal::{
    connect_physical_backend_with_options, AcceleratorDevice, BackendKind, DeviceState, HalError,
    PhysicalBackendOptions,
};
use zervram_scrub::{
    default_chain_file_path, default_quarantine_file_path, hash_tenant_id,
    load_verifying_key_from_file, parse_verifying_key, verify_bundle_json, AppendOnlyChainStore,
    AttestationBundle, AttestationSigner, CertificateStore, Ed25519Signer, QuarantineStore,
    ScrubConfig, ScrubError, ScrubRequest, VerifyPolicy, VramScrubber, VramZeroEngine,
    EXIT_CODE_QUARANTINED,
};

const MAX_RESPONSE_BYTES: usize = 1024 * 1024;

#[derive(Debug, Error)]
enum BinError {
    #[error("HAL error: {0}")]
    Hal(#[from] HalError),

    #[error("scrub error: {0}")]
    Scrub(#[from] ScrubError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("{0}")]
    InvalidInput(String),

    #[error("verification failed: {0}")]
    VerificationFailed(String),

    #[error("daemon quarantined device: {0}")]
    DaemonQuarantined(String),

    #[error("daemon RPC failed: {0}")]
    DaemonRpc(String),
}

#[derive(Debug)]
enum DaemonCallError {
    Unreachable(String),
    Quarantined(String),
    Failed(String),
}

#[derive(Debug, Serialize)]
struct DaemonScrubPayload {
    device: usize,
    allow_mock: bool,
    challenge_nonce: Option<String>,
    tenant_id: Option<String>,
    job_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DaemonErrorPayload {
    #[serde(default)]
    code: String,
    #[serde(default)]
    error: String,
}

#[derive(Debug, Parser)]
#[command(
    name = "zervram-scrub",
    version,
    about = "Post-job GPU VRAM sanitizer & Ed25519 v2 attestation CLI (CUDA PTX / NVML / ROCm)"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Zero-based GPU device ordinal to sanitize (e.g., --device 0).
    #[arg(long, default_value_t = 0)]
    device: usize,

    /// Explicitly authorize software mock-gpu backend when no physical CUDA/ROCm GPU is present.
    #[arg(long, default_value_t = false)]
    allow_mock: bool,

    /// Bypass fail-closed NVML active-process safety gate (active processes are still attested).
    #[arg(long, default_value_t = false)]
    force: bool,

    /// Percentage of free physical VRAM to sweep (1..=100, default: 99).
    #[arg(long)]
    sweep_ratio_pct: Option<u8>,

    /// Persistent PKCS#8 Ed25519 signing key file (auto-created atomically with 0600 on Unix).
    #[arg(long, default_value = "node.key")]
    key_file: PathBuf,

    /// Persistent append-only JSONL hash-chain file (defaults to OS local state dir `chain.jsonl`).
    #[arg(long)]
    chain_file: Option<PathBuf>,

    /// Persistent device quarantine registry JSON file (defaults to sibling `quarantine.json`).
    #[arg(long)]
    quarantine_file: Option<PathBuf>,

    /// Optional verifier/scheduler challenge nonce signed directly into the certificate.
    #[arg(long)]
    nonce: Option<String>,

    /// Output machine-readable signed JSON attestation bundle to stdout.
    #[arg(long, default_value_t = false)]
    json: bool,

    /// Departing tenant identifier (hashed with SHA-256 before attestation).
    #[arg(long, default_value = "epilog-prev-tenant")]
    tenant_id: String,

    /// Departing job or container identifier.
    #[arg(long, default_value = "epilog-job")]
    job_id: String,

    /// Optional file path to write the JSON attestation bundle (for `zervram-scrub verify`).
    #[arg(long)]
    out_cert: Option<PathBuf>,

    /// Route scrub request to a warm local `zervramd` instance (e.g., `http://127.0.0.1:9740`).
    #[arg(long)]
    daemon_url: Option<String>,

    /// Fall back to direct cold CLI scrub ONLY if `--daemon-url` is unreachable.
    #[arg(long, default_value_t = false)]
    fallback_direct: bool,

    /// Optional bearer token file when authenticating to `zervramd`.
    #[arg(long)]
    token_file: Option<PathBuf>,

    /// Optional bearer token string (or `ZERVRAM_DAEMON_TOKEN` env var) for `zervramd`.
    #[arg(long, env = "ZERVRAM_DAEMON_TOKEN", hide_env_values = true)]
    daemon_token: Option<String>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Verify an exported ZeroizationCertificateV2 JSON bundle offline (fail-closed by default).
    Verify {
        /// Path to JSON file written by `zervram-scrub --json` / `--out-cert`.
        cert_path: PathBuf,

        /// Pinned Ed25519 key file (`node.key` PKCS#8 PEM or 64-char hex `.pub`).
        #[arg(long)]
        key_file: Option<PathBuf>,

        /// Pinned 64-character hex Ed25519 verifying key.
        #[arg(long)]
        pubkey: Option<String>,

        /// Directory of pinned trusted `.pub` / `.hex` / `.key` files.
        #[arg(long)]
        trust_dir: Option<PathBuf>,

        /// Optional expected challenge nonce that must match `certificate.challenge_nonce`.
        #[arg(long)]
        expected_nonce: Option<String>,

        /// UNSAFE: allow verifying against an untrusted public key embedded inside the JSON file.
        #[arg(long, default_value_t = false)]
        insecure_trust_embedded_key: bool,
    },

    /// Audit an entire persistent `chain.jsonl` hash chain from genesis to head against pinned keys.
    VerifyChain {
        /// Path to persistent `chain.jsonl` (defaults to OS local state dir).
        #[arg(long)]
        chain_file: Option<PathBuf>,

        /// Pinned Ed25519 key file (`node.key` PKCS#8 PEM or 64-char hex `.pub`).
        #[arg(long)]
        key_file: Option<PathBuf>,

        /// Pinned 64-character hex Ed25519 verifying key.
        #[arg(long)]
        pubkey: Option<String>,

        /// Directory of pinned trusted `.pub` / `.hex` / `.key` files.
        #[arg(long)]
        trust_dir: Option<PathBuf>,
    },
}

fn select_backend(
    allow_mock: bool,
    options: PhysicalBackendOptions,
) -> Result<Arc<dyn AcceleratorDevice>, BinError> {
    match connect_physical_backend_with_options(options) {
        Ok(physical) => Ok(physical),
        Err(phys_err) => {
            if !allow_mock {
                return Err(BinError::Hal(phys_err));
            }
            #[cfg(feature = "mock-gpu")]
            {
                Ok(Arc::new(zervram_hal::MockAcceleratorBackend::new(
                    zervram_hal::MockClusterConfig::default(),
                )))
            }
            #[cfg(not(feature = "mock-gpu"))]
            {
                Err(BinError::InvalidInput(format!(
                    "--allow-mock requested, but binary was compiled without `mock-gpu` feature ({phys_err})"
                )))
            }
        }
    }
}

fn resolve_trusted_keys(
    pubkey: Option<String>,
    key_file: Option<PathBuf>,
    trust_dir: Option<PathBuf>,
) -> Result<Vec<VerifyingKey>, BinError> {
    let mut keys: Vec<VerifyingKey> = Vec::new();
    let mut explicit_vk: Option<VerifyingKey> = None;

    if let Some(pk_hex) = pubkey {
        let vk = parse_verifying_key(&pk_hex)?;
        explicit_vk = Some(vk);
        keys.push(vk);
    }

    if let Some(kf) = key_file {
        let vk = load_verifying_key_from_file(&kf)?;
        if let Some(prev) = explicit_vk {
            if prev != vk {
                return Err(BinError::VerificationFailed(
                    "--pubkey and --key-file disagree on Ed25519 public key identity".to_string(),
                ));
            }
        } else {
            keys.push(vk);
        }
    }

    if let Some(dir) = trust_dir {
        let entries = std::fs::read_dir(&dir).map_err(|err| {
            BinError::VerificationFailed(format!(
                "failed to read --trust-dir '{}': {err}",
                dir.display()
            ))
        })?;
        let mut dir_count: usize = 0;
        for entry_res in entries {
            let entry = entry_res?;
            let path = entry.path();
            if path.is_file() {
                let vk = load_verifying_key_from_file(&path)?;
                if !keys.contains(&vk) {
                    keys.push(vk);
                }
                dir_count = dir_count.saturating_add(1);
            }
        }
        if dir_count == 0 {
            return Err(BinError::VerificationFailed(format!(
                "--trust-dir '{}' contains 0 key files",
                dir.display()
            )));
        }
    }

    Ok(keys)
}

fn load_client_token(
    token_file: Option<&Path>,
    cli_token: Option<&str>,
) -> Result<Option<String>, BinError> {
    if let Some(path) = token_file {
        let raw = std::fs::read_to_string(path)?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(BinError::InvalidInput(format!(
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

fn parse_daemon_authority(raw_url: &str) -> Result<String, DaemonCallError> {
    let trimmed = raw_url.trim().trim_end_matches('/');
    let Some(rest) = trimmed
        .strip_prefix("http://")
        .or_else(|| trimmed.strip_prefix("HTTP://"))
    else {
        return Err(DaemonCallError::Failed(format!(
            "unsupported --daemon-url '{raw_url}' (must start with http://)"
        )));
    };
    if rest.is_empty() || rest.contains('/') || rest.chars().any(char::is_whitespace) {
        return Err(DaemonCallError::Failed(format!(
            "invalid --daemon-url authority '{raw_url}'"
        )));
    }
    Ok(rest.to_string())
}

async fn write_all_tcp(stream: &TcpStream, mut data: &[u8]) -> std::io::Result<()> {
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

async fn read_some_tcp(stream: &TcpStream, buf: &mut [u8]) -> std::io::Result<usize> {
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

async fn call_daemon_scrub(
    daemon_url: &str,
    bearer_token: Option<&str>,
    payload: &DaemonScrubPayload,
) -> Result<AttestationBundle, DaemonCallError> {
    let authority = parse_daemon_authority(daemon_url)?;
    let stream = match tokio::time::timeout(Duration::from_secs(5), TcpStream::connect(&authority))
        .await
    {
        Ok(Ok(s)) => s,
        Ok(Err(err)) => {
            return Err(DaemonCallError::Unreachable(format!(
                "connect to {authority} failed: {err}"
            )));
        }
        Err(_) => {
            return Err(DaemonCallError::Unreachable(format!(
                "timed out connecting to {authority}"
            )));
        }
    };
    let _ = stream.set_nodelay(true);

    let body = serde_json::to_vec(payload)
        .map_err(|e| DaemonCallError::Failed(format!("JSON encode failed: {e}")))?;
    let auth_line = match bearer_token {
        Some(tok) => {
            if tok.contains('\r') || tok.contains('\n') {
                return Err(DaemonCallError::Failed(
                    "bearer token contains invalid newline characters".to_string(),
                ));
            }
            format!("Authorization: Bearer {tok}\r\n")
        }
        None => String::new(),
    };
    let req_head = format!(
        "POST /v1/scrub HTTP/1.1\r\nHost: {authority}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{auth_line}Connection: close\r\n\r\n",
        body.len()
    );

    write_all_tcp(&stream, req_head.as_bytes())
        .await
        .map_err(|e| DaemonCallError::Failed(format!("failed writing request headers: {e}")))?;
    write_all_tcp(&stream, &body)
        .await
        .map_err(|e| DaemonCallError::Failed(format!("failed writing request body: {e}")))?;

    let mut raw: Vec<u8> = Vec::with_capacity(2048);
    let mut buf = [0u8; 4096];
    let mut header_end: Option<usize> = None;
    let mut content_len: Option<usize> = None;

    let read_res = tokio::time::timeout(Duration::from_secs(120), async {
        loop {
            let n = read_some_tcp(&stream, &mut buf)
                .await
                .map_err(|e| DaemonCallError::Failed(format!("read error: {e}")))?;
            if n == 0 {
                break;
            }
            if let Some(slice) = buf.get(..n) {
                raw.extend_from_slice(slice);
            }
            if raw.len() > MAX_RESPONSE_BYTES {
                return Err(DaemonCallError::Failed(
                    "daemon HTTP response exceeded 1 MiB".to_string(),
                ));
            }
            if header_end.is_none() {
                if let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                    header_end = Some(pos);
                    if let Some(hbytes) = raw.get(..pos) {
                        if let Ok(htext) = std::str::from_utf8(hbytes) {
                            for line in htext.split("\r\n").skip(1) {
                                if let Some((k, v)) = line.split_once(':') {
                                    if k.trim().eq_ignore_ascii_case("content-length") {
                                        content_len = v.trim().parse::<usize>().ok();
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if let (Some(hend), Some(clen)) = (header_end, content_len) {
                if raw.len() >= hend.saturating_add(4).saturating_add(clen) {
                    break;
                }
            }
        }
        Ok::<(), DaemonCallError>(())
    })
    .await;

    match read_res {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(e),
        Err(_) => {
            return Err(DaemonCallError::Failed(
                "timed out awaiting daemon /v1/scrub response".to_string(),
            ));
        }
    }

    let hend = header_end.ok_or_else(|| {
        DaemonCallError::Failed("malformed HTTP response from daemon (no header terminator)".to_string())
    })?;
    let hbytes = raw
        .get(..hend)
        .ok_or_else(|| DaemonCallError::Failed("malformed HTTP response headers".to_string()))?;
    let htext = std::str::from_utf8(hbytes)
        .map_err(|_| DaemonCallError::Failed("non-UTF8 HTTP response headers".to_string()))?;
    let status_line = htext
        .split("\r\n")
        .next()
        .ok_or_else(|| DaemonCallError::Failed("missing HTTP status line".to_string()))?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u16>().ok())
        .ok_or_else(|| DaemonCallError::Failed(format!("invalid HTTP status line '{status_line}'")))?;

    let mut resp_body = raw.get(hend.saturating_add(4)..).unwrap_or(&[]).to_vec();
    if let Some(clen) = content_len {
        if resp_body.len() < clen {
            return Err(DaemonCallError::Failed(
                "truncated HTTP response body from daemon".to_string(),
            ));
        }
        resp_body.truncate(clen);
    }

    if status == 200 {
        serde_json::from_slice::<AttestationBundle>(&resp_body).map_err(|e| {
            DaemonCallError::Failed(format!("failed to parse AttestationBundle JSON: {e}"))
        })
    } else if status == 423 {
        let msg = serde_json::from_slice::<DaemonErrorPayload>(&resp_body)
            .map(|p| p.error)
            .unwrap_or_else(|_| String::from_utf8_lossy(&resp_body).into_owned());
        Err(DaemonCallError::Quarantined(msg))
    } else {
        let detail = serde_json::from_slice::<DaemonErrorPayload>(&resp_body)
            .map(|p| format!("[{}] {}", p.code, p.error))
            .unwrap_or_else(|_| String::from_utf8_lossy(&resp_body).into_owned());
        Err(DaemonCallError::Failed(format!(
            "HTTP {status} from {daemon_url}: {detail}"
        )))
    }
}

fn emit_bundle(
    bundle: &AttestationBundle,
    out_cert: Option<&Path>,
    json: bool,
    source_label: &str,
    source_value: &str,
) -> Result<(), BinError> {
    let json_text = serde_json::to_string_pretty(bundle)?;
    if let Some(out_path) = out_cert {
        if let Some(parent) = out_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(out_path, json_text.as_bytes())?;
    }

    if json {
        println!("{json_text}");
    } else {
        println!("ZERVRAM VRAM SANITIZATION VERIFIED (v2)");
        println!("  Schema           : {}", bundle.certificate.schema_version);
        println!("  Backend          : {:?}", bundle.certificate.backend_kind);
        println!(
            "  Device [{}]       : {} ({})",
            bundle.certificate.device_ordinal,
            bundle.certificate.vendor_model,
            bundle.certificate.device_uuid
        );
        println!(
            "  Swept / Free     : {} / {} bytes ({} chunks)",
            bundle.certificate.swept_bytes,
            bundle.certificate.vram_free_bytes,
            bundle.certificate.chunk_count
        );
        println!(
            "  Verification     : {:?} ({} bytes verified, {} canary pages)",
            bundle.certificate.verification_mode,
            bundle.certificate.verified_bytes,
            bundle.certificate.sampled_pages
        );
        println!("  Duration         : {} us", bundle.certificate.duration_us);
        println!("  Prev Cert SHA256 : {}", bundle.certificate.prev_cert_sha256);
        println!("  Cert SHA-256     : {}", bundle.certificate_sha256);
        println!("  {source_label:<16} : {source_value}");
        println!("  Ed25519 PubKey   : {}", bundle.ed25519_public_key_hex);
    }
    Ok(())
}

async fn run_scrub(cli: Cli) -> Result<(), BinError> {
    if let Some(ref daemon_url) = cli.daemon_url {
        let token = load_client_token(cli.token_file.as_deref(), cli.daemon_token.as_deref())?;
        let payload = DaemonScrubPayload {
            device: cli.device,
            allow_mock: cli.allow_mock,
            challenge_nonce: cli.nonce.clone(),
            tenant_id: Some(cli.tenant_id.clone()),
            job_id: Some(cli.job_id.clone()),
        };
        match call_daemon_scrub(daemon_url, token.as_deref(), &payload).await {
            Ok(bundle) => {
                return emit_bundle(
                    &bundle,
                    cli.out_cert.as_deref(),
                    cli.json,
                    "Daemon URL",
                    daemon_url,
                );
            }
            Err(DaemonCallError::Quarantined(msg)) => {
                return Err(BinError::DaemonQuarantined(msg));
            }
            Err(DaemonCallError::Unreachable(msg)) if cli.fallback_direct => {
                eprintln!(
                    "zervram-scrub: warning: daemon {daemon_url} unreachable ({msg}); falling back to direct cold scrub"
                );
            }
            Err(DaemonCallError::Unreachable(msg) | DaemonCallError::Failed(msg)) => {
                return Err(BinError::DaemonRpc(msg));
            }
        }
    }

    let (signer_obj, created_key) = Ed25519Signer::load_or_generate_key_file(&cli.key_file)?;
    let pubkey_hex = signer_obj.verifying_key_hex();
    if created_key {
        eprintln!(
            "zervram-scrub: generated persistent Ed25519 signing key at {} (pubkey={})",
            cli.key_file.display(),
            pubkey_hex
        );
    }

    let chain_path = cli.chain_file.unwrap_or_else(default_chain_file_path);
    let quarantine_path = cli
        .quarantine_file
        .unwrap_or_else(|| default_quarantine_file_path(Some(&chain_path)));

    let phys_opts = PhysicalBackendOptions {
        force_ignore_active_processes: cli.force,
        sweep_free_vram_pct: cli.sweep_ratio_pct,
    };
    let hal = select_backend(cli.allow_mock, phys_opts)?;
    let devices = hal.discover_devices().await?;
    let target = devices.get(cli.device).ok_or_else(|| {
        BinError::InvalidInput(format!(
            "requested --device {} out of range ({} device(s) discovered on {:?})",
            cli.device,
            devices.len(),
            hal.backend_kind()
        ))
    })?;

    let quarantine_store = Arc::new(QuarantineStore::open(&quarantine_path));
    if let Some(rec) = quarantine_store.is_quarantined(target.device_uuid)? {
        return Err(BinError::Scrub(ScrubError::DeviceQuarantined {
            device_uuid: target.device_uuid,
            reason: rec.reason,
        }));
    }

    if hal.backend_kind() == BackendKind::MockCpu && target.state == DeviceState::Available {
        let _ = hal
            .allocate_slice(
                target.device_uuid,
                hash_tenant_id(&cli.tenant_id),
                cli.job_id.clone(),
            )
            .await?;
    }

    let signer: Arc<dyn AttestationSigner> = Arc::new(signer_obj);
    let store: Arc<dyn CertificateStore> = Arc::new(AppendOnlyChainStore::open(&chain_path)?);
    let config = match hal.backend_kind() {
        BackendKind::MockCpu => ScrubConfig::default(),
        _ => ScrubConfig::production(600_000),
    };
    let engine = VramZeroEngine::new(Arc::clone(&hal), signer, store, config)
        .with_quarantine_store(quarantine_store);

    let cert = engine
        .sanitize_on_job_exit(ScrubRequest {
            device_uuid: target.device_uuid,
            job_id: cli.job_id,
            prev_tenant_id: cli.tenant_id,
            explicit_range: None,
            challenge_nonce: cli.nonce,
        })
        .await?;

    let cert_sha = cert.certificate_sha256_hex()?;
    let bundle = AttestationBundle {
        ed25519_public_key_hex: pubkey_hex,
        certificate_sha256: cert_sha,
        certificate: cert,
    };

    emit_bundle(
        &bundle,
        cli.out_cert.as_deref(),
        cli.json,
        "Chain File",
        &chain_path.display().to_string(),
    )
}

fn run_verify(
    cert_path: PathBuf,
    key_file: Option<PathBuf>,
    pubkey: Option<String>,
    trust_dir: Option<PathBuf>,
    expected_nonce: Option<String>,
    insecure_trust_embedded_key: bool,
) -> Result<(), BinError> {
    let raw_json = std::fs::read_to_string(&cert_path)?;
    let trusted_keys = resolve_trusted_keys(pubkey, key_file, trust_dir)?;

    let policy = VerifyPolicy {
        trusted_keys,
        allow_insecure_embedded_key: insecure_trust_embedded_key,
        expected_nonce,
    };

    let summary = verify_bundle_json(&raw_json, &policy)?;
    if summary.used_insecure_embedded_key {
        eprintln!(
            "WARNING: --insecure-trust-embedded-key enabled; certificate authenticity is NOT anchored to a pinned key!"
        );
    }

    println!("VERIFICATION PASSED");
    println!("  Certificate ID   : {}", summary.cert_id);
    println!("  Backend          : {:?}", summary.backend_kind);
    println!(
        "  Device [{}]       : {} ({})",
        summary.device_ordinal, summary.vendor_model, summary.device_uuid
    );
    println!(
        "  Verified Bytes   : {} / {} ({:?}, {} pages)",
        summary.verified_bytes,
        summary.swept_bytes,
        summary.verification_mode,
        summary.sampled_pages
    );
    println!("  Duration         : {} us", summary.duration_us);
    println!("  Cert SHA-256     : {}", summary.computed_cert_sha256);
    println!("  Ed25519 PubKey   : {}", summary.ed25519_public_key_hex);
    Ok(())
}

async fn run_verify_chain(
    chain_file: Option<PathBuf>,
    key_file: Option<PathBuf>,
    pubkey: Option<String>,
    trust_dir: Option<PathBuf>,
) -> Result<(), BinError> {
    let trusted_keys = resolve_trusted_keys(pubkey, key_file, trust_dir)?;
    if trusted_keys.is_empty() {
        return Err(BinError::VerificationFailed(
            "verify-chain requires a pinned trust anchor: provide --pubkey <hex>, --key-file <path>, or --trust-dir <dir>"
                .to_string(),
        ));
    }

    let path = chain_file.unwrap_or_else(default_chain_file_path);
    if !Path::new(&path).exists() {
        return Err(BinError::VerificationFailed(format!(
            "chain file '{}' does not exist",
            path.display()
        )));
    }

    let store = AppendOnlyChainStore::open(&path)?;
    let summary = store.verify_full_chain_with_keys(&trusted_keys).await?;

    if summary.total_certificates == 0 {
        return Err(BinError::VerificationFailed(format!(
            "chain file '{}' contains 0 certificates",
            path.display()
        )));
    }
    if !summary.chain_intact || !summary.all_signatures_valid {
        return Err(BinError::VerificationFailed(format!(
            "chain audit FAILED (total={}, chain_intact={}, all_signatures_valid={})",
            summary.total_certificates, summary.chain_intact, summary.all_signatures_valid
        )));
    }

    println!("CHAIN VERIFICATION PASSED");
    println!("  Chain File       : {}", path.display());
    println!("  Certificates     : {}", summary.total_certificates);
    println!("  Chain Intact     : {}", summary.chain_intact);
    println!("  Signatures Valid : {}", summary.all_signatures_valid);
    println!("  Head SHA-256     : {}", summary.head_sha256);
    Ok(())
}

fn exit_code_for_error(err: &BinError) -> ExitCode {
    match err {
        BinError::Scrub(ScrubError::ResidualDataDetected { .. })
        | BinError::Scrub(ScrubError::DeviceQuarantined { .. })
        | BinError::Hal(HalError::HardwareFaultQuarantine { .. })
        | BinError::DaemonQuarantined(_) => ExitCode::from(EXIT_CODE_QUARANTINED),
        _ => ExitCode::FAILURE,
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Some(Commands::Verify {
            cert_path,
            key_file,
            pubkey,
            trust_dir,
            expected_nonce,
            insecure_trust_embedded_key,
        }) => run_verify(
            cert_path,
            key_file,
            pubkey,
            trust_dir,
            expected_nonce,
            insecure_trust_embedded_key,
        ),
        Some(Commands::VerifyChain {
            chain_file,
            key_file,
            pubkey,
            trust_dir,
        }) => run_verify_chain(chain_file, key_file, pubkey, trust_dir).await,
        None => run_scrub(cli).await,
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            let code = exit_code_for_error(&err);
            eprintln!("error: {err}");
            code
        }
    }
}