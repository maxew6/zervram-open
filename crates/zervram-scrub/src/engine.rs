use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use chrono::Utc;
use ed25519_dalek::VerifyingKey;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use zervram_hal::{
    AcceleratorDevice, BackendKind, DeviceState, HalError, MemoryRange, VerificationMode,
    VRAM_PAGE_SIZE_BYTES,
};

use crate::certificate::{
    hash_tenant_id, CertificateVerificationResult, ChainAuditSummary, ZeroizationCertificate,
    ZeroizationCertificateV2, CERTIFICATE_SCHEMA_V2, GENESIS_PREV_CERT_SHA256,
};
use crate::error::ScrubError;
use crate::signer::AttestationSigner;
use crate::store::{CertificateStore, QuarantineRecord, QuarantineStore};
use crate::verifier::{verify_zeroized_pages, SamplingStrategy};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrubConfig {
    /// Target upper bound in microseconds (`3_000` for mock, `600_000` for cold CLI GPU scrub).
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
            warn_only_on_sla_breach: true,
            sampling_strategy: SamplingStrategy::DeterministicAllMapped,
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
    /// Optional verifier/scheduler replay-protection challenge nonce signed into the certificate.
    pub challenge_nonce: Option<String>,
}

#[async_trait]
pub trait VramScrubber: Send + Sync + 'static {
    /// Full post-job zeroization, GPU/host read-back verification, Ed25519 v2 attestation,
    /// SHA-256 hash-chain append, persistent quarantine on fault, and `mark_available` gate.
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

struct MintParams {
    device_uuid: Uuid,
    device_ordinal: u32,
    pci_bus_id: String,
    vendor_model: String,
    backend_kind: BackendKind,
    tenant_prev_hash: String,
    vram_total_bytes: u64,
    vram_free_bytes: u64,
    swept_bytes: u64,
    chunk_count: u32,
    chunk_table_sha256: String,
    verification_mode: VerificationMode,
    verified_bytes: u64,
    sampled_pages: u64,
    active_processes_before: u32,
    challenge_nonce: Option<String>,
    duration_us: u64,
    verified_zero: bool,
}

pub struct VramZeroEngine {
    hal: Arc<dyn AcceleratorDevice>,
    signer: Arc<dyn AttestationSigner>,
    store: Arc<dyn CertificateStore>,
    quarantine_store: Option<Arc<QuarantineStore>>,
    config: ScrubConfig,
}

impl fmt::Debug for VramZeroEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VramZeroEngine")
            .field("backend_kind", &self.hal.backend_kind())
            .field("verifying_key_hex", &self.signer.verifying_key_hex())
            .field("has_quarantine_store", &self.quarantine_store.is_some())
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
            quarantine_store: None,
            config,
        }
    }

    pub fn with_quarantine_store(mut self, quarantine_store: Arc<QuarantineStore>) -> Self {
        self.quarantine_store = Some(quarantine_store);
        self
    }

    async fn mint_and_append_certificate(
        &self,
        params: MintParams,
    ) -> Result<ZeroizationCertificate, ScrubError> {
        let signer = Arc::clone(&self.signer);
        let cert_id = Uuid::now_v7();
        let timestamp_utc = Utc::now();

        let (cert, _cert_sha256) = self
            .store
            .append_signed(Box::new(move |prev_cert_sha256| {
                let mut cert = ZeroizationCertificateV2 {
                    schema_version: CERTIFICATE_SCHEMA_V2.to_string(),
                    cert_id,
                    device_uuid: params.device_uuid,
                    device_ordinal: params.device_ordinal,
                    pci_bus_id: params.pci_bus_id,
                    vendor_model: params.vendor_model,
                    backend_kind: params.backend_kind,
                    tenant_prev_hash: params.tenant_prev_hash,
                    vram_total_bytes: params.vram_total_bytes,
                    vram_free_bytes: params.vram_free_bytes,
                    swept_bytes: params.swept_bytes,
                    chunk_count: params.chunk_count,
                    chunk_table_sha256: params.chunk_table_sha256,
                    verification_mode: params.verification_mode,
                    verified_bytes: params.verified_bytes,
                    sampled_pages: params.sampled_pages,
                    active_processes_before: params.active_processes_before,
                    challenge_nonce: params.challenge_nonce,
                    duration_us: params.duration_us,
                    verified_zero: params.verified_zero,
                    timestamp_utc,
                    prev_cert_sha256,
                    ed25519_signature_hex: String::new(),
                };
                let digest = cert.canonical_signing_digest()?;
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
        // Fail closed if device was persistently quarantined on an earlier invocation.
        if let Some(ref q_store) = self.quarantine_store {
            if let Some(rec) = q_store.is_quarantined(request.device_uuid)? {
                return Err(ScrubError::DeviceQuarantined {
                    device_uuid: request.device_uuid,
                    reason: rec.reason,
                });
            }
        }

        let tenant_prev_hash = hash_tenant_id(&request.prev_tenant_id);
        let descriptor = self.hal.get_device(request.device_uuid).await?;

        if let DeviceState::Quarantined { ref reason, .. } = descriptor.state {
            return Err(ScrubError::DeviceQuarantined {
                device_uuid: request.device_uuid,
                reason: reason.clone(),
            });
        }

        let requested_range = match request.explicit_range {
            Some(explicit) => {
                if matches!(descriptor.state, DeviceState::Allocated { .. }) {
                    let _ = self.hal.release_to_dirty(request.device_uuid).await?;
                }
                explicit
            }
            None => self.hal.release_to_dirty(request.device_uuid).await?,
        };

        let op_started = Instant::now();
        let receipt = match self
            .hal
            .zeroize_range_async(request.device_uuid, requested_range)
            .await
        {
            Ok(receipt) => receipt,
            Err(HalError::HardwareFaultQuarantine {
                device_uuid,
                reason,
            }) => {
                if let Some(ref q_store) = self.quarantine_store {
                    let _ = q_store.record_quarantine(QuarantineRecord {
                        device_uuid,
                        pci_bus_id: descriptor.pci_bus_id.clone(),
                        vendor_model: descriptor.vendor_model_label.clone(),
                        reason: reason.clone(),
                        quarantined_at: Utc::now(),
                        failed_cert_id: None,
                    });
                }
                return Err(ScrubError::DeviceQuarantined {
                    device_uuid,
                    reason,
                });
            }
            Err(err) => return Err(ScrubError::Hal(err)),
        };

        let actual_range = receipt.range;

        let report = verify_zeroized_pages(
            self.hal.as_ref(),
            request.device_uuid,
            actual_range,
            self.config.sampling_strategy,
        )
        .await?;

        let total_duration_us =
            u64::try_from(op_started.elapsed().as_micros()).unwrap_or(receipt.duration_us);

        let gpu_clean = receipt.gpu_reduction.map_or(true, |r| r.is_clean());
        let all_zero = report.all_zero && gpu_clean;

        let sampled_pages_u64 = u64::try_from(report.sampled_pages).unwrap_or(u64::MAX);
        let verified_bytes = match (receipt.verification_mode, receipt.gpu_reduction) {
            (VerificationMode::GpuPtxFullReduction, Some(red)) => {
                if red.words_scanned == red.expected_words {
                    red.verified_bytes
                } else {
                    red.words_scanned.saturating_mul(8)
                }
            }
            _ => sampled_pages_u64.saturating_mul(u64::from(VRAM_PAGE_SIZE_BYTES)),
        };

        if !all_zero {
            let (page_addr, non_zero_bytes, fault_detail) = if !report.all_zero {
                let addr = report
                    .first_dirty_page_addr
                    .unwrap_or(actual_range.start_addr);
                let nzb = report.first_dirty_non_zero_bytes.max(1);
                (
                    addr,
                    nzb,
                    format!(
                        "residual non-zero VRAM data at page {addr:#x} ({nzb} bytes) after job {}",
                        request.job_id
                    ),
                )
            } else if let Some(red) = receipt.gpu_reduction {
                let nzb = usize::try_from(red.non_zero_or_mask.count_ones().max(1)).unwrap_or(1);
                (
                    actual_range.start_addr,
                    nzb,
                    format!(
                        "GPU PTX full reduction failed (or_mask={:#x}, words_scanned={}/{}) after job {}",
                        red.non_zero_or_mask, red.words_scanned, red.expected_words, request.job_id
                    ),
                )
            } else {
                (
                    actual_range.start_addr,
                    1,
                    format!("verification failed after job {}", request.job_id),
                )
            };

            if self.config.quarantine_on_residual_data {
                self.hal
                    .quarantine_device(request.device_uuid, fault_detail.clone())
                    .await?;
            }

            let failed_cert = self
                .mint_and_append_certificate(MintParams {
                    device_uuid: request.device_uuid,
                    device_ordinal: descriptor.device_ordinal,
                    pci_bus_id: descriptor.pci_bus_id.clone(),
                    vendor_model: descriptor.vendor_model_label.clone(),
                    backend_kind: receipt.backend_kind,
                    tenant_prev_hash,
                    vram_total_bytes: receipt.vram_total_bytes,
                    vram_free_bytes: receipt.vram_free_bytes,
                    swept_bytes: receipt.swept_bytes,
                    chunk_count: receipt.chunk_count,
                    chunk_table_sha256: receipt.chunk_table_sha256,
                    verification_mode: receipt.verification_mode,
                    verified_bytes,
                    sampled_pages: sampled_pages_u64,
                    active_processes_before: receipt.active_processes_before,
                    challenge_nonce: request.challenge_nonce,
                    duration_us: total_duration_us,
                    verified_zero: false,
                })
                .await?;

            if self.config.quarantine_on_residual_data {
                if let Some(ref q_store) = self.quarantine_store {
                    q_store.record_quarantine(QuarantineRecord {
                        device_uuid: request.device_uuid,
                        pci_bus_id: descriptor.pci_bus_id,
                        vendor_model: descriptor.vendor_model_label,
                        reason: fault_detail,
                        quarantined_at: Utc::now(),
                        failed_cert_id: Some(failed_cert.cert_id),
                    })?;
                }
            }

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
            .mint_and_append_certificate(MintParams {
                device_uuid: request.device_uuid,
                device_ordinal: descriptor.device_ordinal,
                pci_bus_id: descriptor.pci_bus_id,
                vendor_model: descriptor.vendor_model_label,
                backend_kind: receipt.backend_kind,
                tenant_prev_hash,
                vram_total_bytes: receipt.vram_total_bytes,
                vram_free_bytes: receipt.vram_free_bytes,
                swept_bytes: receipt.swept_bytes,
                chunk_count: receipt.chunk_count,
                chunk_table_sha256: receipt.chunk_table_sha256,
                verification_mode: receipt.verification_mode,
                verified_bytes,
                sampled_pages: sampled_pages_u64,
                active_processes_before: receipt.active_processes_before,
                challenge_nonce: request.challenge_nonce,
                duration_us: total_duration_us,
                verified_zero: true,
            })
            .await?;

        self.hal
            .mark_available(request.device_uuid, cert.cert_id, total_duration_us)
            .await?;

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
            Some(prev) => prev.certificate_sha256_hex()?,
            None => GENESIS_PREV_CERT_SHA256.to_string(),
        };
        let chain_link_valid = cert.prev_cert_sha256 == expected_prev;

        Ok(CertificateVerificationResult {
            cert_id: cert.cert_id,
            signature_valid,
            chain_link_valid,
            verified_zero_flag: cert.verified_zero,
            computed_cert_sha256: cert.certificate_sha256_hex()?,
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
    use crate::store::{AppendOnlyChainStore, InMemoryCertificateStore};
    use zervram_hal::{
        AcceleratorModel, DeviceDescriptor, MockAcceleratorBackend, MockClusterConfig, PageSample,
        ZeroizeReceipt, READBACK_POISON_BYTE,
    };

    /// Simulates a broken driver or no-op DMA shim that leaves `0xAA` poison bytes untouched.
    #[derive(Debug)]
    struct PoisonNoopReadbackBackend {
        inner: MockAcceleratorBackend,
    }

    #[async_trait]
    impl AcceleratorDevice for PoisonNoopReadbackBackend {
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
            self.inner.zeroize_range_async(device_uuid, range).await
        }
        async fn read_back_pages(
            &self,
            _device_uuid: Uuid,
            page_addrs: &[u64],
        ) -> Result<Vec<PageSample>, HalError> {
            Ok(page_addrs
                .iter()
                .map(|&page_addr| PageSample {
                    page_addr,
                    data: Box::new([READBACK_POISON_BYTE; VRAM_PAGE_SIZE_BYTES as usize]),
                })
                .collect())
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
    async fn mints_hash_chained_v2_certificates_and_rejects_backend_tamper(
    ) -> Result<(), ScrubError> {
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
            ScrubConfig::default(),
        );

        let devices = hal.discover_devices().await?;
        let dev = devices
            .iter()
            .find(|d| d.model == AcceleratorModel::NvidiaH100Sxm)
            .ok_or(HalError::UnknownHardwareModel {
                raw: "nvidia_h100_sxm".to_string(),
            })?;

        let _ = hal
            .allocate_slice(dev.device_uuid, hash_tenant_id("tenant-1"), "job-1".to_string())
            .await?;
        let c0 = engine
            .sanitize_on_job_exit(ScrubRequest {
                device_uuid: dev.device_uuid,
                job_id: "job-1".to_string(),
                prev_tenant_id: "tenant-1".to_string(),
                explicit_range: None,
                challenge_nonce: Some("sched-nonce-1".to_string()),
            })
            .await?;
        assert!(c0.verified_zero);
        assert_eq!(c0.schema_version, CERTIFICATE_SCHEMA_V2);
        assert_eq!(c0.backend_kind, BackendKind::MockCpu);
        assert_eq!(c0.challenge_nonce.as_deref(), Some("sched-nonce-1"));
        assert_eq!(c0.prev_cert_sha256, GENESIS_PREV_CERT_SHA256);

        let _ = hal
            .allocate_slice(dev.device_uuid, hash_tenant_id("tenant-2"), "job-2".to_string())
            .await?;
        let c1 = engine
            .sanitize_on_job_exit(ScrubRequest {
                device_uuid: dev.device_uuid,
                job_id: "job-2".to_string(),
                prev_tenant_id: "tenant-2".to_string(),
                explicit_range: None,
                challenge_nonce: None,
            })
            .await?;
        assert_eq!(c1.prev_cert_sha256, c0.certificate_sha256_hex()?);

        let mut tampered = c1.clone();
        tampered.backend_kind = BackendKind::CudaNative;
        assert!(matches!(
            tampered.verify_signature(&signer.verifying_key()),
            Err(ScrubError::InvalidSignature { .. })
        ));
        Ok(())
    }

    #[tokio::test]
    async fn poison_0xaa_noop_readback_triggers_persistent_quarantine_and_blocks_restart(
    ) -> Result<(), ScrubError> {
        let dir = std::env::temp_dir().join(format!("zervram_poison_test_{}", Uuid::new_v4()));
        let q_path = dir.join("quarantine.json");
        let chain_path = dir.join("chain.jsonl");

        let faulty = PoisonNoopReadbackBackend {
            inner: MockAcceleratorBackend::new(MockClusterConfig::default()),
        };
        let hal: Arc<dyn AcceleratorDevice> = Arc::new(faulty);
        let signer = Arc::new(Ed25519Signer::generate_ephemeral());
        let store = Arc::new(AppendOnlyChainStore::open(&chain_path)?);
        let q_store = Arc::new(QuarantineStore::open(&q_path));

        let engine = VramZeroEngine::new(
            Arc::clone(&hal),
            Arc::clone(&signer) as Arc<dyn AttestationSigner>,
            Arc::clone(&store) as Arc<dyn CertificateStore>,
            ScrubConfig::default(),
        )
        .with_quarantine_store(Arc::clone(&q_store));

        let devices = hal.discover_devices().await?;
        let target = devices.first().ok_or(HalError::UnknownHardwareModel {
            raw: "empty-mock".to_string(),
        })?;

        let _ = hal
            .allocate_slice(
                target.device_uuid,
                hash_tenant_id("tenant-poison"),
                "job-poison".to_string(),
            )
            .await?;

        let err = engine
            .sanitize_on_job_exit(ScrubRequest {
                device_uuid: target.device_uuid,
                job_id: "job-poison".to_string(),
                prev_tenant_id: "tenant-poison".to_string(),
                explicit_range: None,
                challenge_nonce: None,
            })
            .await;

        assert!(matches!(
            err,
            Err(ScrubError::ResidualDataDetected {
                non_zero_bytes: 4096,
                ..
            })
        ));
        assert!(q_store.is_quarantined(target.device_uuid)?.is_some());

        // Fresh engine simulating next CLI run MUST fail closed with DeviceQuarantined.
        let fresh_hal: Arc<dyn AcceleratorDevice> =
            Arc::new(MockAcceleratorBackend::new(MockClusterConfig::default()));
        let fresh_engine = VramZeroEngine::new(
            fresh_hal,
            Arc::clone(&signer) as Arc<dyn AttestationSigner>,
            Arc::new(AppendOnlyChainStore::open(&chain_path)?),
            ScrubConfig::default(),
        )
        .with_quarantine_store(Arc::new(QuarantineStore::open(&q_path)));

        let blocked = fresh_engine
            .sanitize_on_job_exit(ScrubRequest {
                device_uuid: target.device_uuid,
                job_id: "job-after-quarantine".to_string(),
                prev_tenant_id: "tenant-next".to_string(),
                explicit_range: None,
                challenge_nonce: None,
            })
            .await;
        assert!(matches!(blocked, Err(ScrubError::DeviceQuarantined { .. })));

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    #[tokio::test]
    async fn zero_sampled_pages_fails_closed_instead_of_vacuous_true() -> Result<(), ScrubError> {
        let mock = MockAcceleratorBackend::new(MockClusterConfig::default());
        let hal: Arc<dyn AcceleratorDevice> = Arc::new(mock);
        let devices = hal.discover_devices().await?;
        let target = devices.first().ok_or(HalError::UnknownHardwareModel {
            raw: "empty-mock".to_string(),
        })?;
        let range = hal
            .allocate_slice(
                target.device_uuid,
                hash_tenant_id("t"),
                "j".to_string(),
            )
            .await?;
        let _ = hal.release_to_dirty(target.device_uuid).await?;
        let _ = hal.zeroize_range_async(target.device_uuid, range).await?;

        let err = verify_zeroized_pages(
            hal.as_ref(),
            target.device_uuid,
            range,
            SamplingStrategy::StratifiedSpotSample {
                random_interior_pages: 0,
                include_boundary_pages: false,
            },
        )
        .await;
        assert!(matches!(err, Err(ScrubError::EmptyVerificationSample { .. })));
        Ok(())
    }
}