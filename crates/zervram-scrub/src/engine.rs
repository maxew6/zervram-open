use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use chrono::Utc;
use ed25519_dalek::VerifyingKey;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use zervram_hal::{AcceleratorDevice, DeviceState, MemoryRange};

use crate::certificate::{
    hash_tenant_id, CertificateVerificationResult, ChainAuditSummary, ZeroizationCertificate,
    GENESIS_PREV_CERT_SHA256,
};
use crate::error::ScrubError;
use crate::signer::AttestationSigner;
use crate::store::CertificateStore;
use crate::verifier::{verify_zeroized_pages, SamplingStrategy};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrubConfig {
    /// Target upper bound in microseconds (`3_000` in mock/demo mode, `5_000` in production).
    pub sla_target_us: u64,
    /// When `true`, exceeding `sla_target_us` emits a warning log instead of failing attestation.
    pub warn_only_on_sla_breach: bool,
    pub sampling_strategy: SamplingStrategy,
    pub quarantine_on_residual_data: bool,
}

impl Default for ScrubConfig {
    fn default() -> Self {
        Self {
            sla_target_us: 3_000,
            warn_only_on_sla_breach: true,
            sampling_strategy: SamplingStrategy::DeterministicAllMapped,
            quarantine_on_residual_data: true,
        }
    }
}

impl ScrubConfig {
    pub fn production(sla_target_us: u64) -> Self {
        Self {
            sla_target_us,
            warn_only_on_sla_breach: false,
            sampling_strategy: SamplingStrategy::StratifiedSpotSample {
                random_interior_pages: 64,
                include_boundary_pages: true,
            },
            quarantine_on_residual_data: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrubRequest {
    pub device_uuid: Uuid,
    pub job_id: String,
    pub prev_tenant_id: String,
    /// Optional explicit range; when `None`, obtained from `AcceleratorDevice::release_to_dirty`.
    pub explicit_range: Option<MemoryRange>,
}

#[async_trait]
pub trait VramScrubber: Send + Sync + 'static {
    /// Full post-job zeroization, 4KB read-back verification, Ed25519 attestation,
    /// SHA-256 hash-chain append, and `mark_available` release gate.
    async fn sanitize_on_job_exit(
        &self,
        request: ScrubRequest,
    ) -> Result<ZeroizationCertificate, ScrubError>;

    async fn verify_certificate(
        &self,
        cert_id: Uuid,
    ) -> Result<CertificateVerificationResult, ScrubError>;

    async fn verify_chain(&self) -> Result<ChainAuditSummary, ScrubError>;

    async fn recent_certificates(
        &self,
        limit: usize,
    ) -> Result<Vec<ZeroizationCertificate>, ScrubError>;

    fn verifying_key_hex(&self) -> String;
}

pub struct VramZeroEngine {
    hal: Arc<dyn AcceleratorDevice>,
    signer: Arc<dyn AttestationSigner>,
    store: Arc<dyn CertificateStore>,
    config: ScrubConfig,
}

impl fmt::Debug for VramZeroEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VramZeroEngine")
            .field("backend_kind", &self.hal.backend_kind())
            .field("verifying_key_hex", &self.signer.verifying_key_hex())
            .field("config", &self.config)
            .finish()
    }
}

impl VramZeroEngine {
    pub fn new(
        hal: Arc<dyn AcceleratorDevice>,
        signer: Arc<dyn AttestationSigner>,
        store: Arc<dyn CertificateStore>,
        config: ScrubConfig,
    ) -> Self {
        Self {
            hal,
            signer,
            store,
            config,
        }
    }

    async fn mint_and_append_certificate(
        &self,
        device_uuid: Uuid,
        vendor_model: String,
        tenant_prev_hash: String,
        range: MemoryRange,
        duration_us: u64,
        verified_zero: bool,
    ) -> Result<ZeroizationCertificate, ScrubError> {
        let signer = Arc::clone(&self.signer);
        let cert_id = Uuid::now_v7();
        let timestamp_utc = Utc::now();

        let (cert, _cert_sha256) = self
            .store
            .append_signed(Box::new(move |prev_cert_sha256| {
                let mut cert = ZeroizationCertificate {
                    cert_id,
                    device_uuid,
                    vendor_model,
                    tenant_prev_hash,
                    start_addr: range.start_addr,
                    byte_len: range.byte_len,
                    duration_us,
                    verified_zero,
                    timestamp_utc,
                    prev_cert_sha256,
                    ed25519_signature_hex: String::new(),
                };
                let digest = cert.canonical_signing_digest();
                let signature = signer.sign_digest(&digest)?;
                cert.ed25519_signature_hex = hex::encode(signature);
                Ok(cert)
            }))
            .await?;

        Ok(cert)
    }
}

#[async_trait]
impl VramScrubber for VramZeroEngine {
    async fn sanitize_on_job_exit(
        &self,
        request: ScrubRequest,
    ) -> Result<ZeroizationCertificate, ScrubError> {
        let tenant_prev_hash = hash_tenant_id(&request.prev_tenant_id);
        let descriptor = self.hal.get_device(request.device_uuid).await?;

        let range = match request.explicit_range {
            Some(explicit) => {
                if matches!(descriptor.state, DeviceState::Allocated { .. }) {
                    let _ = self.hal.release_to_dirty(request.device_uuid).await?;
                }
                explicit
            }
            None => self.hal.release_to_dirty(request.device_uuid).await?,
        };

        let op_started = Instant::now();
        let receipt = self
            .hal
            .zeroize_range_async(request.device_uuid, range)
            .await?;

        let report = verify_zeroized_pages(
            self.hal.as_ref(),
            request.device_uuid,
            range,
            self.config.sampling_strategy,
        )
        .await?;

        let total_duration_us =
            u64::try_from(op_started.elapsed().as_micros()).unwrap_or(receipt.duration_us);

        if !report.all_zero {
            let page_addr = report.first_dirty_page_addr.unwrap_or(range.start_addr);
            let non_zero_bytes = report.first_dirty_non_zero_bytes;

            if self.config.quarantine_on_residual_data {
                let reason = format!(
                    "residual non-zero VRAM data at page {page_addr:#x} ({non_zero_bytes} bytes) after job {}",
                    request.job_id
                );
                self.hal
                    .quarantine_device(request.device_uuid, reason)
                    .await?;
            }

            // Record signed forensic failure certificate in the immutable hash chain.
            let _failed_cert = self
                .mint_and_append_certificate(
                    request.device_uuid,
                    descriptor.vendor_model_label,
                    tenant_prev_hash,
                    range,
                    total_duration_us,
                    false,
                )
                .await?;

            return Err(ScrubError::ResidualDataDetected {
                device_uuid: request.device_uuid,
                page_addr,
                non_zero_bytes,
            });
        }

        if total_duration_us > self.config.sla_target_us {
            if self.config.warn_only_on_sla_breach {
                tracing::warn!(
                    device_uuid = %request.device_uuid,
                    duration_us = total_duration_us,
                    sla_target_us = self.config.sla_target_us,
                    "VRAM zeroization exceeded target SLA budget"
                );
            } else {
                return Err(ScrubError::SlaBudgetExceeded {
                    device_uuid: request.device_uuid,
                    duration_us: total_duration_us,
                    sla_max_us: self.config.sla_target_us,
                });
            }
        }

        let cert = self
            .mint_and_append_certificate(
                request.device_uuid,
                descriptor.vendor_model_label,
                tenant_prev_hash,
                range,
                total_duration_us,
                true,
            )
            .await?;

        self.hal
            .mark_available(request.device_uuid, cert.cert_id, total_duration_us)
            .await?;

        tracing::debug!(
            cert_id = %cert.cert_id,
            device_uuid = %cert.device_uuid,
            duration_us = cert.duration_us,
            sampled_pages = report.sampled_pages,
            "minted verified Ed25519 ZeroizationCertificate"
        );

        Ok(cert)
    }

    async fn verify_certificate(
        &self,
        cert_id: Uuid,
    ) -> Result<CertificateVerificationResult, ScrubError> {
        let Some((cert, predecessor)) = self.store.get_with_predecessor(cert_id).await? else {
            return Err(ScrubError::CertificateNotFound { cert_id });
        };

        let verifying_key = VerifyingKey::from_bytes(&self.signer.verifying_key_bytes())
            .map_err(|err| ScrubError::SigningFailure(err.to_string()))?;

        let signature_valid = cert.verify_signature(&verifying_key).is_ok();
        let expected_prev = match predecessor {
            Some(prev) => prev.certificate_sha256_hex(),
            None => GENESIS_PREV_CERT_SHA256.to_string(),
        };
        let chain_link_valid = cert.prev_cert_sha256 == expected_prev;

        Ok(CertificateVerificationResult {
            cert_id: cert.cert_id,
            signature_valid,
            chain_link_valid,
            verified_zero_flag: cert.verified_zero,
            computed_cert_sha256: cert.certificate_sha256_hex(),
            ed25519_public_key_hex: self.signer.verifying_key_hex(),
        })
    }

    async fn verify_chain(&self) -> Result<ChainAuditSummary, ScrubError> {
        self.store
            .verify_full_chain(&self.signer.verifying_key_bytes())
            .await
    }

    async fn recent_certificates(
        &self,
        limit: usize,
    ) -> Result<Vec<ZeroizationCertificate>, ScrubError> {
        self.store.list_recent(limit).await
    }

    fn verifying_key_hex(&self) -> String {
        self.signer.verifying_key_hex()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signer::Ed25519Signer;
    use crate::store::InMemoryCertificateStore;
    use zervram_hal::{
        AcceleratorModel, BackendKind, DeviceDescriptor, HalError, MockAcceleratorBackend,
        MockClusterConfig, PageSample, ZeroizeReceipt,
    };

    /// Fault-injecting HAL decorator simulating a defective HBM bank leaving a single 0xFF byte
    /// after zeroization to verify fail-closed detection and permanent quarantine.
    #[derive(Debug)]
    struct ResidualFaultBackend {
        inner: MockAcceleratorBackend,
    }

    #[async_trait]
    impl AcceleratorDevice for ResidualFaultBackend {
        fn backend_kind(&self) -> BackendKind {
            self.inner.backend_kind()
        }

        async fn discover_devices(&self) -> Result<Vec<DeviceDescriptor>, HalError> {
            self.inner.discover_devices().await
        }

        async fn get_device(&self, device_uuid: Uuid) -> Result<DeviceDescriptor, HalError> {
            self.inner.get_device(device_uuid).await
        }

        async fn allocate_slice(
            &self,
            device_uuid: Uuid,
            tenant_id_hash: String,
            job_id: String,
        ) -> Result<MemoryRange, HalError> {
            self.inner
                .allocate_slice(device_uuid, tenant_id_hash, job_id)
                .await
        }

        async fn release_to_dirty(&self, device_uuid: Uuid) -> Result<MemoryRange, HalError> {
            self.inner.release_to_dirty(device_uuid).await
        }

        async fn zeroize_range_async(
            &self,
            device_uuid: Uuid,
            range: MemoryRange,
        ) -> Result<ZeroizeReceipt, HalError> {
            let receipt = self.inner.zeroize_range_async(device_uuid, range).await?;
            self.inner
                .inject_residual_byte_for_test(device_uuid, range.start_addr, 64, 0xFF)
                .await?;
            Ok(receipt)
        }

        async fn read_back_pages(
            &self,
            device_uuid: Uuid,
            page_addrs: &[u64],
        ) -> Result<Vec<PageSample>, HalError> {
            self.inner.read_back_pages(device_uuid, page_addrs).await
        }

        async fn mapped_verification_pages(
            &self,
            device_uuid: Uuid,
            range: MemoryRange,
        ) -> Result<Option<Vec<u64>>, HalError> {
            self.inner
                .mapped_verification_pages(device_uuid, range)
                .await
        }

        async fn mark_available(
            &self,
            device_uuid: Uuid,
            cert_id: Uuid,
            scrub_duration_us: u64,
        ) -> Result<(), HalError> {
            self.inner
                .mark_available(device_uuid, cert_id, scrub_duration_us)
                .await
        }

        async fn quarantine_device(
            &self,
            device_uuid: Uuid,
            reason: String,
        ) -> Result<(), HalError> {
            self.inner.quarantine_device(device_uuid, reason).await
        }
    }

    #[tokio::test]
    async fn mints_hash_chained_ed25519_certificates_and_verifies_chain() -> Result<(), ScrubError> {
        let mock = MockAcceleratorBackend::new(MockClusterConfig {
            backing_pages_per_device: 32,
            ..MockClusterConfig::default()
        });
        let hal: Arc<dyn AcceleratorDevice> = Arc::new(mock);
        let signer = Arc::new(Ed25519Signer::generate_ephemeral());
        let store = Arc::new(InMemoryCertificateStore::new());
        let engine = VramZeroEngine::new(
            Arc::clone(&hal),
            Arc::clone(&signer) as Arc<dyn AttestationSigner>,
            Arc::clone(&store) as Arc<dyn CertificateStore>,
            ScrubConfig {
                sla_target_us: 3_000,
                warn_only_on_sla_breach: true,
                sampling_strategy: SamplingStrategy::DeterministicAllMapped,
                quarantine_on_residual_data: true,
            },
        );

        let devices = hal.discover_devices().await?;
        let targets = [
            AcceleratorModel::NvidiaH100Sxm,
            AcceleratorModel::NvidiaB200Blackwell,
            AcceleratorModel::FuriosaRngdSriovSlice,
        ];

        let mut minted = Vec::new();
        for (idx, model) in targets.iter().enumerate() {
            let dev = devices
                .iter()
                .find(|d| d.model == *model)
                .ok_or(HalError::UnknownHardwareModel {
                    raw: format!("{model:?}"),
                })?;

            let tenant_id = format!("kyc-tenant-{idx}");
            let job_id = format!("job-train-{idx}");
            let _ = hal
                .allocate_slice(dev.device_uuid, hash_tenant_id(&tenant_id), job_id.clone())
                .await?;

            let cert = engine
                .sanitize_on_job_exit(ScrubRequest {
                    device_uuid: dev.device_uuid,
                    job_id,
                    prev_tenant_id: tenant_id.clone(),
                    explicit_range: None,
                })
                .await?;

            assert!(cert.verified_zero);
            assert_eq!(cert.device_uuid, dev.device_uuid);
            assert_eq!(cert.tenant_prev_hash, hash_tenant_id(&tenant_id));
            assert_eq!(cert.ed25519_signature_hex.len(), 128);
            #[cfg(not(debug_assertions))]
            assert!(cert.duration_us < 3_000);

            let updated_dev = hal.get_device(dev.device_uuid).await?;
            assert_eq!(updated_dev.state, DeviceState::Available);
            assert_eq!(updated_dev.last_cert_id, Some(cert.cert_id));

            minted.push(cert);
        }

        // Verify SHA-256 hash chain linkage: GENESIS -> cert[0] -> cert[1] -> cert[2]
        let c0 = minted.first().ok_or(ScrubError::CertificateNotFound {
            cert_id: Uuid::nil(),
        })?;
        let c1 = minted.get(1).ok_or(ScrubError::CertificateNotFound {
            cert_id: Uuid::nil(),
        })?;
        let c2 = minted.get(2).ok_or(ScrubError::CertificateNotFound {
            cert_id: Uuid::nil(),
        })?;

        assert_eq!(c0.prev_cert_sha256, GENESIS_PREV_CERT_SHA256);
        assert_eq!(c1.prev_cert_sha256, c0.certificate_sha256_hex());
        assert_eq!(c2.prev_cert_sha256, c1.certificate_sha256_hex());

        for cert in &minted {
            let result = engine.verify_certificate(cert.cert_id).await?;
            assert!(result.signature_valid);
            assert!(result.chain_link_valid);
            assert!(result.verified_zero_flag);
            assert_eq!(result.ed25519_public_key_hex, signer.verifying_key_hex());
        }

        let summary = engine.verify_chain().await?;
        assert_eq!(summary.total_certificates, 3);
        assert!(summary.all_signatures_valid);
        assert!(summary.chain_intact);
        assert_eq!(summary.head_sha256, c2.certificate_sha256_hex());

        // Tampering with any field must invalidate Ed25519 signature verification
        let mut tampered = c2.clone();
        tampered.duration_us = tampered.duration_us.saturating_add(1);
        assert!(matches!(
            tampered.verify_signature(&signer.verifying_key()),
            Err(ScrubError::InvalidSignature { .. })
        ));

        Ok(())
    }

    #[tokio::test]
    async fn residual_non_zero_byte_quarantines_device_and_records_failed_cert(
    ) -> Result<(), ScrubError> {
        let faulty = ResidualFaultBackend {
            inner: MockAcceleratorBackend::new(MockClusterConfig::default()),
        };
        let hal: Arc<dyn AcceleratorDevice> = Arc::new(faulty);
        let signer = Arc::new(Ed25519Signer::generate_ephemeral());
        let store = Arc::new(InMemoryCertificateStore::new());
        let engine = VramZeroEngine::new(
            Arc::clone(&hal),
            Arc::clone(&signer) as Arc<dyn AttestationSigner>,
            Arc::clone(&store) as Arc<dyn CertificateStore>,
            ScrubConfig::default(),
        );

        let devices = hal.discover_devices().await?;
        let b200 = devices
            .iter()
            .find(|d| d.model == AcceleratorModel::NvidiaB200Blackwell)
            .ok_or(HalError::UnknownHardwareModel {
                raw: "nvidia_b200_blackwell".to_string(),
            })?;

        let _ = hal
            .allocate_slice(
                b200.device_uuid,
                hash_tenant_id("kyc-fault-tenant"),
                "job-fault".to_string(),
            )
            .await?;

        let err = engine
            .sanitize_on_job_exit(ScrubRequest {
                device_uuid: b200.device_uuid,
                job_id: "job-fault".to_string(),
                prev_tenant_id: "kyc-fault-tenant".to_string(),
                explicit_range: None,
            })
            .await;

        assert!(matches!(
            err,
            Err(ScrubError::ResidualDataDetected {
                non_zero_bytes: 1,
                ..
            })
        ));

        let desc = hal.get_device(b200.device_uuid).await?;
        assert!(matches!(desc.state, DeviceState::Quarantined { .. }));

        // Forensic failure certificate was still signed and chained
        let recent = engine.recent_certificates(10).await?;
        assert_eq!(recent.len(), 1);
        let failed_cert = recent.first().ok_or(ScrubError::CertificateNotFound {
            cert_id: Uuid::nil(),
        })?;
        assert!(!failed_cert.verified_zero);

        let verify = engine.verify_certificate(failed_cert.cert_id).await?;
        assert!(verify.signature_valid);
        assert!(verify.chain_link_valid);
        assert!(!verify.verified_zero_flag);
        Ok(())
    }

    #[tokio::test]
    async fn stratified_spot_sampling_verifies_zeroized_hbm_pages() -> Result<(), ScrubError> {
        let mock = MockAcceleratorBackend::new(MockClusterConfig::default());
        let hal: Arc<dyn AcceleratorDevice> = Arc::new(mock);
        let signer = Arc::new(Ed25519Signer::generate_ephemeral());
        let store = Arc::new(InMemoryCertificateStore::new());
        let engine = VramZeroEngine::new(
            Arc::clone(&hal),
            Arc::clone(&signer) as Arc<dyn AttestationSigner>,
            Arc::clone(&store) as Arc<dyn CertificateStore>,
            ScrubConfig {
                sla_target_us: 5_000,
                warn_only_on_sla_breach: true,
                sampling_strategy: SamplingStrategy::StratifiedSpotSample {
                    random_interior_pages: 32,
                    include_boundary_pages: true,
                },
                quarantine_on_residual_data: true,
            },
        );

        let devices = hal.discover_devices().await?;
        let mi300x = devices
            .iter()
            .find(|d| d.model == AcceleratorModel::AmdInstinctMi300x)
            .ok_or(HalError::UnknownHardwareModel {
                raw: "amd_instinct_mi300x".to_string(),
            })?;

        let _ = hal
            .allocate_slice(
                mi300x.device_uuid,
                hash_tenant_id("kyc-amd-tenant"),
                "job-mi300x".to_string(),
            )
            .await?;

        let cert = engine
            .sanitize_on_job_exit(ScrubRequest {
                device_uuid: mi300x.device_uuid,
                job_id: "job-mi300x".to_string(),
                prev_tenant_id: "kyc-amd-tenant".to_string(),
                explicit_range: None,
            })
            .await?;

        assert!(cert.verified_zero);
        assert_eq!(hal.get_device(mi300x.device_uuid).await?.state, DeviceState::Available);
        Ok(())
    }
}
