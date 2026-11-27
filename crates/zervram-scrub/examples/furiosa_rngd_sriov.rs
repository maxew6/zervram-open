#![forbid(unsafe_code)]

use std::sync::Arc;

use zervram_hal::{
    AcceleratorDevice, AcceleratorModel, MockAcceleratorBackend, MockClusterConfig,
};
use zervram_scrub::{
    hash_tenant_id, AttestationSigner, CertificateStore, Ed25519Signer, InMemoryCertificateStore,
    ScrubConfig, ScrubRequest, VramScrubber, VramZeroEngine,
};

/// Demonstrates software `mock-gpu` simulation of 8x FuriosaAI RNGD SR-IOV slices (1,024 TPP each)
/// undergoing post-job zeroization and Ed25519 v2 hash-chained attestation.
///
/// Note: Native FuriosaAI FFI bindings (`furiosa-native`) are currently `experimental-stub`;
/// this example runs deterministically on CPU via `MockAcceleratorBackend`.
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let hal: Arc<dyn AcceleratorDevice> = Arc::new(MockAcceleratorBackend::new(MockClusterConfig {
        h100_count: 0,
        b200_count: 0,
        mi300x_count: 0,
        blackhole_p150_count: 0,
        furiosa_rngd_physical_cards: 1,
        ..MockClusterConfig::default()
    }));

    let signer = Arc::new(Ed25519Signer::generate_ephemeral());
    let store = Arc::new(InMemoryCertificateStore::new());
    let engine = VramZeroEngine::new(
        Arc::clone(&hal),
        Arc::clone(&signer) as Arc<dyn AttestationSigner>,
        Arc::clone(&store) as Arc<dyn CertificateStore>,
        ScrubConfig::default(),
    );

    let slices = hal.discover_devices().await?;
    println!(
        "Discovered {} simulated FuriosaAI RNGD SR-IOV slices (mock-gpu):",
        slices.len()
    );

    for (idx, slice) in slices.iter().enumerate() {
        assert_eq!(slice.model, AcceleratorModel::FuriosaRngdSriovSlice);
        let tenant = format!("tenant-inference-{idx}");
        let job = format!("rngd-slice-job-{idx}");

        let _ = hal
            .allocate_slice(slice.device_uuid, hash_tenant_id(&tenant), job.clone())
            .await?;

        let cert = engine
            .sanitize_on_job_exit(ScrubRequest {
                device_uuid: slice.device_uuid,
                job_id: job,
                prev_tenant_id: tenant,
                explicit_range: None,
                challenge_nonce: Some(format!("slice-nonce-{idx}")),
            })
            .await?;

        println!(
            "  slice[{}] uuid={} verified_bytes={} mode={:?} prev_sha256={}..",
            idx,
            cert.device_uuid,
            cert.verified_bytes,
            cert.verification_mode,
            cert.prev_cert_sha256.get(..12).unwrap_or("")
        );
    }

    let summary = engine.verify_chain().await?;
    println!(
        "Chain audit complete: total={} intact={} signatures_valid={} head={}",
        summary.total_certificates,
        summary.chain_intact,
        summary.all_signatures_valid,
        summary.head_sha256
    );
    Ok(())
}