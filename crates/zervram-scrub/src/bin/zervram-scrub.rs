#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use ed25519_dalek::VerifyingKey;
use thiserror::Error;
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

async fn run_scrub(cli: Cli) -> Result<(), BinError> {
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

    // In explicit --allow-mock mode, simulate a departing tenant job tainting VRAM so the full
    // dirty -> zeroize -> 4 KiB 0x00 verification pipeline executes end-to-end.
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

    let json_text = serde_json::to_string_pretty(&bundle)?;

    if let Some(out_path) = &cli.out_cert {
        if let Some(parent) = out_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(out_path, json_text.as_bytes())?;
    }

    if cli.json {
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
        println!("  Chain File       : {}", chain_path.display());
        println!("  Ed25519 PubKey   : {}", bundle.ed25519_public_key_hex);
    }

    Ok(())
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
        | BinError::Hal(HalError::HardwareFaultQuarantine { .. }) => {
            ExitCode::from(EXIT_CODE_QUARANTINED)
        }
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