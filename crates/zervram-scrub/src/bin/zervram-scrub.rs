#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use ed25519_dalek::VerifyingKey;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use zervram_hal::{connect_physical_backend, AcceleratorDevice, BackendKind, DeviceState};
use zervram_scrub::{
    hash_tenant_id, AttestationSigner, CertificateStore, Ed25519Signer, InMemoryCertificateStore,
    ScrubConfig, ScrubRequest, VramScrubber, VramZeroEngine, ZeroizationCertificate,
};

#[derive(Debug, Error)]
enum BinError {
    #[error("HAL error: {0}")]
    Hal(#[from] zervram_hal::HalError),

    #[error("scrub error: {0}")]
    Scrub(#[from] zervram_scrub::ScrubError),

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
    about = "Post-job GPU VRAM sanitizer & Ed25519 attestation CLI (CUDA / ROCm dynamic FFI)"
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

    /// Persistent PKCS#8 Ed25519 signing key file (auto-created on first run if absent).
    #[arg(long, default_value = "node.key")]
    key_file: PathBuf,

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
    /// Verify an exported ZeroizationCertificate JSON file offline.
    Verify {
        /// Path to JSON file written by `zervram-scrub --json` or a raw `ZeroizationCertificate`.
        cert_path: PathBuf,

        /// Optional persistent Ed25519 key file (`node.key`) to cross-check public key identity.
        #[arg(long)]
        key_file: Option<PathBuf>,

        /// Optional 64-character hex Ed25519 verifying key override.
        #[arg(long)]
        pubkey: Option<String>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct AttestationBundle {
    schema_version: String,
    backend_kind: BackendKind,
    device_ordinal: usize,
    ed25519_public_key_hex: String,
    certificate_sha256: String,
    certificate: ZeroizationCertificate,
}

fn select_backend(allow_mock: bool) -> Result<Arc<dyn AcceleratorDevice>, BinError> {
    match connect_physical_backend() {
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

    let hal = select_backend(cli.allow_mock)?;
    let devices = hal.discover_devices().await?;
    let target = devices.get(cli.device).ok_or_else(|| {
        BinError::InvalidInput(format!(
            "requested --device {} out of range ({} device(s) discovered on {:?})",
            cli.device,
            devices.len(),
            hal.backend_kind()
        ))
    })?;

    // In explicit --allow-mock mode, simulate a departing tenant job tainting VRAM so the full
    // LeftoverLocals dirty -> zeroize -> 4KB 0x00 verification pipeline executes end-to-end.
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
    let store: Arc<dyn CertificateStore> = Arc::new(InMemoryCertificateStore::new());
    let config = match hal.backend_kind() {
        BackendKind::MockCpu => ScrubConfig::default(),
        _ => ScrubConfig::production(50_000),
    };
    let engine = VramZeroEngine::new(Arc::clone(&hal), signer, store, config);

    let cert = engine
        .sanitize_on_job_exit(ScrubRequest {
            device_uuid: target.device_uuid,
            job_id: cli.job_id,
            prev_tenant_id: cli.tenant_id,
            explicit_range: None,
        })
        .await?;

    let bundle = AttestationBundle {
        schema_version: "zervram.scrub.v1".to_string(),
        backend_kind: hal.backend_kind(),
        device_ordinal: cli.device,
        ed25519_public_key_hex: pubkey_hex,
        certificate_sha256: cert.certificate_sha256_hex(),
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
        println!("ZERVRAM VRAM SANITIZATION VERIFIED");
        println!("  Backend          : {:?}", bundle.backend_kind);
        println!("  Device [{}]       : {} ({})", bundle.device_ordinal, bundle.certificate.vendor_model, bundle.certificate.device_uuid);
        println!("  Swept VRAM       : {} bytes @ {:#x}", bundle.certificate.byte_len, bundle.certificate.start_addr);
        println!("  Duration         : {} us", bundle.certificate.duration_us);
        println!("  Verified 0x00    : {}", bundle.certificate.verified_zero);
        println!("  Cert SHA-256     : {}", bundle.certificate_sha256);
        println!("  Ed25519 PubKey   : {}", bundle.ed25519_public_key_hex);
        println!("  Ed25519 Sig      : {}", bundle.certificate.ed25519_signature_hex);
    }

    Ok(())
}

fn parse_verifying_key(hex_str: &str) -> Result<VerifyingKey, BinError> {
    let raw = hex::decode(hex_str.trim())
        .map_err(|err| BinError::VerificationFailed(format!("invalid public key hex: {err}")))?;
    let bytes: [u8; 32] = raw.try_into().map_err(|_| {
        BinError::VerificationFailed("Ed25519 public key must be exactly 32 bytes (64 hex chars)".to_string())
    })?;
    VerifyingKey::from_bytes(&bytes)
        .map_err(|err| BinError::VerificationFailed(format!("invalid Ed25519 public key: {err}")))
}

fn run_verify(
    cert_path: PathBuf,
    key_file: Option<PathBuf>,
    pubkey_override: Option<String>,
) -> Result<(), BinError> {
    let raw_json = std::fs::read_to_string(&cert_path)?;

    let (cert, embedded_pubkey, embedded_hash) =
        if let Ok(bundle) = serde_json::from_str::<AttestationBundle>(&raw_json) {
            (
                bundle.certificate,
                Some(bundle.ed25519_public_key_hex),
                Some(bundle.certificate_sha256),
            )
        } else {
            let cert: ZeroizationCertificate = serde_json::from_str(&raw_json)?;
            (cert, None, None)
        };

    let explicit_pubkey = if let Some(pk) = pubkey_override {
        Some(pk)
    } else if let Some(kf) = key_file {
        let pem = std::fs::read_to_string(&kf)?;
        let signer = Ed25519Signer::from_pkcs8_pem(&pem)?;
        Some(signer.verifying_key_hex())
    } else if embedded_pubkey.is_none() {
        let default_key = PathBuf::from("node.key");
        if default_key.exists() {
            let pem = std::fs::read_to_string(&default_key)?;
            let signer = Ed25519Signer::from_pkcs8_pem(&pem)?;
            Some(signer.verifying_key_hex())
        } else {
            None
        }
    } else {
        None
    };

    if let (Some(emb), Some(exp)) = (&embedded_pubkey, &explicit_pubkey) {
        if !emb.eq_ignore_ascii_case(exp) {
            return Err(BinError::VerificationFailed(format!(
                "public key mismatch: certificate bundle embedded {emb}, key source has {exp}"
            )));
        }
    }

    let final_pubkey_hex = explicit_pubkey.or(embedded_pubkey).ok_or_else(|| {
        BinError::VerificationFailed(
            "no Ed25519 public key available; provide --key-file <node.key> or --pubkey <hex>"
                .to_string(),
        )
    })?;

    let verifying_key = parse_verifying_key(&final_pubkey_hex)?;
    cert.verify_signature(&verifying_key).map_err(|err| {
        BinError::VerificationFailed(format!("Ed25519 signature check failed: {err}"))
    })?;

    let computed_sha = cert.certificate_sha256_hex();
    if let Some(expected_sha) = embedded_hash {
        if !expected_sha.eq_ignore_ascii_case(&computed_sha) {
            return Err(BinError::VerificationFailed(format!(
                "certificate_sha256 mismatch: expected {expected_sha}, computed {computed_sha}"
            )));
        }
    }

    if !cert.verified_zero {
        return Err(BinError::VerificationFailed(format!(
            "certificate {} signature is valid, but verified_zero=false (residual VRAM fault recorded)",
            cert.cert_id
        )));
    }

    println!("VERIFICATION PASSED");
    println!("  Certificate ID   : {}", cert.cert_id);
    println!("  Device           : {} ({})", cert.vendor_model, cert.device_uuid);
    println!("  Verified 0x00    : true");
    println!("  Duration         : {} us", cert.duration_us);
    println!("  Cert SHA-256     : {computed_sha}");
    println!("  Ed25519 PubKey   : {final_pubkey_hex}");
    Ok(())
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Some(Commands::Verify {
            cert_path,
            key_file,
            pubkey,
        }) => run_verify(cert_path, key_file, pubkey),
        None => run_scrub(cli).await,
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}