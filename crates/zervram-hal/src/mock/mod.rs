pub mod topology;
pub mod vram_pool;

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use rand_chacha::ChaCha8Rng;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::backend::AcceleratorDevice;
use crate::error::HalError;
use crate::mock::topology::{build_mock_cluster, next_deterministic_uuid};
use crate::model::{
    BackendKind, DeviceDescriptor, DeviceState, MemoryRange, PageSample, ZeroizeReceipt,
};

pub use vram_pool::SimulatedVramPool;

#[derive(Debug, Clone)]
pub struct MockClusterConfig {
    /// Deterministic PRNG seed for reproducible demo & `--headless-check` runs.
    pub seed: u64,
    /// Count of real 4KB pages backed in host RAM per simulated device slice (default: 128 pages = 512 KiB).
    /// Zeroing 128 real 4KB pages via `zeroize::Zeroize` executes in <3ms on standard laptops.
    pub backing_pages_per_device: usize,
    /// Simulated `LeftoverLocals` dirty-page retention percentage range upon job exit (default: 12..=18).
    pub leftover_dirty_pct_min: u8,
    pub leftover_dirty_pct_max: u8,
    /// Heterogeneous cluster counts instantiated at startup.
    pub h100_count: usize,
    pub b200_count: usize,
    pub mi300x_count: usize,
    pub blackhole_p150_count: usize,
    /// Each physical FuriosaAI RNGD card expands into 8 independently schedulable 1,024-TPP SR-IOV slices.
    pub furiosa_rngd_physical_cards: usize,
}

impl Default for MockClusterConfig {
    fn default() -> Self {
        Self {
            seed: 2026_1116,
            backing_pages_per_device: 128,
            leftover_dirty_pct_min: 12,
            leftover_dirty_pct_max: 18,
            h100_count: 4,
            b200_count: 4,
            mi300x_count: 2,
            blackhole_p150_count: 2,
            furiosa_rngd_physical_cards: 1, // 8x SR-IOV slices (8 x 1,024 = 8,192 TPP)
        }
    }
}

#[derive(Debug)]
pub struct MockDeviceRecord {
    pub descriptor: DeviceDescriptor,
    pub vram_pool: SimulatedVramPool,
    pub(crate) rng: ChaCha8Rng,
}

#[derive(Debug, Clone)]
pub struct MockAcceleratorBackend {
    pub(crate) config: MockClusterConfig,
    pub(crate) devices: Arc<RwLock<BTreeMap<Uuid, MockDeviceRecord>>>,
}

impl MockAcceleratorBackend {
    pub fn new(mut config: MockClusterConfig) -> Self {
        config.backing_pages_per_device = config.backing_pages_per_device.max(1);
        config.leftover_dirty_pct_min = config.leftover_dirty_pct_min.min(100);
        config.leftover_dirty_pct_max = config
            .leftover_dirty_pct_max
            .clamp(config.leftover_dirty_pct_min, 100);

        let devices = build_mock_cluster(&config);
        Self {
            config,
            devices: Arc::new(RwLock::new(devices)),
        }
    }

    pub fn config(&self) -> &MockClusterConfig {
        &self.config
    }

    /// Returns the exact count of currently non-zero mapped 4KB pages on `device_uuid`.
    pub async fn dirty_page_count(&self, device_uuid: Uuid) -> Result<usize, HalError> {
        let guard = self.devices.read().await;
        let record = guard
            .get(&device_uuid)
            .ok_or(HalError::DeviceNotFound { device_uuid })?;
        Ok(record.vram_pool.dirty_page_count())
    }

    /// Fault-injection hook allowing verification & quarantine tests to inject a residual
    /// non-zero byte into a specific mapped 4KB page frame after zeroization.
    pub async fn inject_residual_byte_for_test(
        &self,
        device_uuid: Uuid,
        page_addr: u64,
        byte_offset: usize,
        value: u8,
    ) -> Result<(), HalError> {
        let mut guard = self.devices.write().await;
        let record = guard
            .get_mut(&device_uuid)
            .ok_or(HalError::DeviceNotFound { device_uuid })?;
        record
            .vram_pool
            .inject_byte(device_uuid, page_addr, byte_offset, value)
    }
}

#[async_trait]
impl AcceleratorDevice for MockAcceleratorBackend {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::MockCpu
    }

    async fn discover_devices(&self) -> Result<Vec<DeviceDescriptor>, HalError> {
        let guard = self.devices.read().await;
        let mut descriptors: Vec<DeviceDescriptor> = guard
            .values()
            .map(|rec| rec.descriptor.clone())
            .collect();
        descriptors.sort_by(|a, b| {
            a.model
                .cmp(&b.model)
                .then_with(|| a.node_name.cmp(&b.node_name))
                .then_with(|| a.pci_bus_id.cmp(&b.pci_bus_id))
                .then_with(|| a.device_uuid.cmp(&b.device_uuid))
        });
        Ok(descriptors)
    }

    async fn get_device(&self, device_uuid: Uuid) -> Result<DeviceDescriptor, HalError> {
        let guard = self.devices.read().await;
        let record = guard
            .get(&device_uuid)
            .ok_or(HalError::DeviceNotFound { device_uuid })?;
        Ok(record.descriptor.clone())
    }

    async fn allocate_slice(
        &self,
        device_uuid: Uuid,
        tenant_id_hash: String,
        job_id: String,
    ) -> Result<MemoryRange, HalError> {
        let mut guard = self.devices.write().await;
        let record = guard
            .get_mut(&device_uuid)
            .ok_or(HalError::DeviceNotFound { device_uuid })?;

        if !matches!(record.descriptor.state, DeviceState::Available) {
            return Err(HalError::InvalidDeviceState {
                device_uuid,
                expected: "available",
                actual: record.descriptor.state.status_label().to_string(),
            });
        }

        let allocation_id = next_deterministic_uuid(&mut record.rng);
        record
            .vram_pool
            .taint_for_tenant(&mut record.rng, &tenant_id_hash, &job_id);
        record.descriptor.state = DeviceState::Allocated {
            allocation_id,
            tenant_id_hash,
            job_id: job_id.clone(),
            allocated_at: Utc::now(),
        };

        tracing::debug!(
            %device_uuid,
            %allocation_id,
            %job_id,
            model = %record.descriptor.vendor_model_label,
            "allocated mock accelerator slice"
        );

        Ok(record.vram_pool.active_range)
    }

    async fn release_to_dirty(&self, device_uuid: Uuid) -> Result<MemoryRange, HalError> {
        let mut guard = self.devices.write().await;
        let record = guard
            .get_mut(&device_uuid)
            .ok_or(HalError::DeviceNotFound { device_uuid })?;

        let last_tenant_hash = match &record.descriptor.state {
            DeviceState::Allocated { tenant_id_hash, .. } => tenant_id_hash.clone(),
            other => {
                return Err(HalError::InvalidDeviceState {
                    device_uuid,
                    expected: "allocated",
                    actual: other.status_label().to_string(),
                });
            }
        };

        let estimated_dirty_ratio_pct = record.vram_pool.simulate_leftover_locals_on_release(
            &mut record.rng,
            self.config.leftover_dirty_pct_min,
            self.config.leftover_dirty_pct_max,
        );

        record.descriptor.state = DeviceState::DirtyPendingScrub {
            last_tenant_hash,
            released_at: Utc::now(),
            estimated_dirty_ratio_pct,
        };

        tracing::debug!(
            %device_uuid,
            estimated_dirty_ratio_pct,
            "released mock accelerator slice to dirty_pending_scrub"
        );

        Ok(record.vram_pool.active_range)
    }

    async fn zeroize_range_async(
        &self,
        device_uuid: Uuid,
        range: MemoryRange,
    ) -> Result<ZeroizeReceipt, HalError> {
        let mut guard = self.devices.write().await;
        let record = guard
            .get_mut(&device_uuid)
            .ok_or(HalError::DeviceNotFound { device_uuid })?;

        match &record.descriptor.state {
            DeviceState::DirtyPendingScrub { .. }
            | DeviceState::Scrubbing { .. }
            | DeviceState::Available => {}
            other => {
                return Err(HalError::InvalidDeviceState {
                    device_uuid,
                    expected: "dirty_pending_scrub | scrubbing | available",
                    actual: other.status_label().to_string(),
                });
            }
        }

        if !record.vram_pool.active_range.contains_range(&range) {
            return Err(HalError::InvalidMemoryRange {
                start_addr: range.start_addr,
                byte_len: range.byte_len,
                page_size: range.page_size_bytes,
            });
        }

        record.descriptor.state = DeviceState::Scrubbing {
            started_at: Utc::now(),
        };

        let (duration_us, pages_zeroed) = record.vram_pool.zeroize_range(range)?;

        tracing::debug!(
            %device_uuid,
            duration_us,
            pages_zeroed,
            "zeroized mock accelerator VRAM range"
        );

        Ok(ZeroizeReceipt {
            device_uuid,
            range,
            duration_us,
            pages_zeroed,
            backend_kind: BackendKind::MockCpu,
        })
    }

    async fn read_back_pages(
        &self,
        device_uuid: Uuid,
        page_addrs: &[u64],
    ) -> Result<Vec<PageSample>, HalError> {
        let guard = self.devices.read().await;
        let record = guard
            .get(&device_uuid)
            .ok_or(HalError::DeviceNotFound { device_uuid })?;
        record.vram_pool.read_pages(device_uuid, page_addrs)
    }

    async fn mapped_verification_pages(
        &self,
        device_uuid: Uuid,
        range: MemoryRange,
    ) -> Result<Option<Vec<u64>>, HalError> {
        let guard = self.devices.read().await;
        let record = guard
            .get(&device_uuid)
            .ok_or(HalError::DeviceNotFound { device_uuid })?;
        let pages = record.vram_pool.mapped_pages_in_range(range)?;
        Ok(Some(pages))
    }

    async fn mark_available(
        &self,
        device_uuid: Uuid,
        cert_id: Uuid,
        scrub_duration_us: u64,
    ) -> Result<(), HalError> {
        let mut guard = self.devices.write().await;
        let record = guard
            .get_mut(&device_uuid)
            .ok_or(HalError::DeviceNotFound { device_uuid })?;

        if !matches!(record.descriptor.state, DeviceState::Scrubbing { .. }) {
            return Err(HalError::InvalidDeviceState {
                device_uuid,
                expected: "scrubbing",
                actual: record.descriptor.state.status_label().to_string(),
            });
        }

        record.descriptor.state = DeviceState::Available;
        record.descriptor.last_cert_id = Some(cert_id);
        record.descriptor.last_scrub_duration_us = Some(scrub_duration_us);

        tracing::debug!(
            %device_uuid,
            %cert_id,
            scrub_duration_us,
            "marked mock accelerator slice available after verified zeroization"
        );

        Ok(())
    }

    async fn quarantine_device(
        &self,
        device_uuid: Uuid,
        reason: String,
    ) -> Result<(), HalError> {
        let mut guard = self.devices.write().await;
        let record = guard
            .get_mut(&device_uuid)
            .ok_or(HalError::DeviceNotFound { device_uuid })?;

        tracing::warn!(%device_uuid, %reason, "quarantining mock accelerator device");

        record.descriptor.state = DeviceState::Quarantined {
            reason,
            quarantined_at: Utc::now(),
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::model::{AcceleratorModel, AcceleratorVendor, VRAM_PAGE_SIZE_BYTES};

    #[tokio::test]
    async fn discovers_all_five_heterogeneous_archetypes_with_exact_cluster_tpp(
    ) -> Result<(), HalError> {
        let backend = MockAcceleratorBackend::new(MockClusterConfig::default());
        assert_eq!(backend.backend_kind(), BackendKind::MockCpu);

        let devices = backend.discover_devices().await?;
        // 4 H100 + 4 B200 + 2 MI300X + 2 Blackhole p150 + 8 FuriosaAI RNGD SR-IOV slices = 20
        assert_eq!(devices.len(), 20);

        let h100s = devices
            .iter()
            .filter(|d| d.model == AcceleratorModel::NvidiaH100Sxm)
            .count();
        let b200s = devices
            .iter()
            .filter(|d| d.model == AcceleratorModel::NvidiaB200Blackwell)
            .count();
        let mi300xs = devices
            .iter()
            .filter(|d| d.model == AcceleratorModel::AmdInstinctMi300x)
            .count();
        let blackholes = devices
            .iter()
            .filter(|d| d.model == AcceleratorModel::TenstorrentBlackholeP150)
            .count();
        let rngd_slices: Vec<&DeviceDescriptor> = devices
            .iter()
            .filter(|d| d.model == AcceleratorModel::FuriosaRngdSriovSlice)
            .collect();

        assert_eq!(h100s, 4);
        assert_eq!(b200s, 4);
        assert_eq!(mi300xs, 2);
        assert_eq!(blackholes, 2);
        assert_eq!(rngd_slices.len(), 8);

        let parent_uuids: BTreeSet<Option<Uuid>> =
            rngd_slices.iter().map(|d| d.sriov_parent_uuid).collect();
        assert_eq!(parent_uuids.len(), 1);
        assert!(parent_uuids.iter().all(Option::is_some));

        let slice_indices: BTreeSet<Option<u8>> =
            rngd_slices.iter().map(|d| d.sriov_slice_index).collect();
        assert_eq!(slice_indices.len(), 8);
        for idx in 0u8..8 {
            assert!(slice_indices.contains(&Some(idx)));
        }
        assert!(rngd_slices
            .iter()
            .all(|d| d.vendor == AcceleratorVendor::FuriosaAi && d.tpp_rating == 1_024));

        // 4*15,832 + 4*36,000 + 2*20,880 + 2*9,400 + 8*1,024 = 276,080 TPP
        let total_cluster_tpp: u64 = devices.iter().map(|d| d.tpp_rating).sum();
        assert_eq!(total_cluster_tpp, 276_080);

        let backend_same_seed = MockAcceleratorBackend::new(MockClusterConfig::default());
        assert_eq!(devices, backend_same_seed.discover_devices().await?);
        Ok(())
    }

    #[tokio::test]
    async fn full_vram_sanitization_lifecycle_across_all_five_archetypes() -> Result<(), HalError> {
        let backend = MockAcceleratorBackend::new(MockClusterConfig::default());
        let devices = backend.discover_devices().await?;

        let archetypes = [
            AcceleratorModel::NvidiaH100Sxm,
            AcceleratorModel::NvidiaB200Blackwell,
            AcceleratorModel::AmdInstinctMi300x,
            AcceleratorModel::TenstorrentBlackholeP150,
            AcceleratorModel::FuriosaRngdSriovSlice,
        ];

        for model in archetypes {
            let device = devices
                .iter()
                .find(|d| d.model == model)
                .ok_or(HalError::UnknownHardwareModel {
                    raw: format!("{model:?}"),
                })?;

            let range = backend
                .allocate_slice(
                    device.device_uuid,
                    "sha256-tenant-alpha-prev".to_string(),
                    format!("job-{model:?}"),
                )
                .await?;
            assert_eq!(backend.dirty_page_count(device.device_uuid).await?, 128);

            let dirty_range = backend.release_to_dirty(device.device_uuid).await?;
            assert_eq!(dirty_range, range);

            let reported_pct = match backend.get_device(device.device_uuid).await?.state {
                DeviceState::DirtyPendingScrub {
                    estimated_dirty_ratio_pct,
                    ..
                } => estimated_dirty_ratio_pct,
                other => {
                    return Err(HalError::InvalidDeviceState {
                        device_uuid: device.device_uuid,
                        expected: "dirty_pending_scrub",
                        actual: other.status_label().to_string(),
                    });
                }
            };
            assert!((12..=18).contains(&reported_pct));

            let mapped_pages = backend
                .mapped_verification_pages(device.device_uuid, dirty_range)
                .await?
                .ok_or(HalError::InvalidMemoryRange {
                    start_addr: dirty_range.start_addr,
                    byte_len: dirty_range.byte_len,
                    page_size: dirty_range.page_size_bytes,
                })?;
            assert_eq!(mapped_pages.len(), 128);

            let pre_scrub = backend
                .read_back_pages(device.device_uuid, &mapped_pages)
                .await?;
            let dirty_before = pre_scrub.iter().filter(|s| !s.is_all_zero()).count();
            let expected_dirty = 128usize
                .saturating_mul(usize::from(reported_pct))
                .div_ceil(100);
            assert_eq!(dirty_before, expected_dirty);

            let receipt = backend
                .zeroize_range_async(device.device_uuid, dirty_range)
                .await?;
            assert_eq!(receipt.backend_kind, BackendKind::MockCpu);
            assert_eq!(receipt.pages_zeroed, 128);
            assert_eq!(receipt.range, dirty_range);
            #[cfg(not(debug_assertions))]
            assert!(receipt.duration_us < 3_000);

            assert!(matches!(
                backend.get_device(device.device_uuid).await?.state,
                DeviceState::Scrubbing { .. }
            ));

            let post_scrub = backend
                .read_back_pages(device.device_uuid, &mapped_pages)
                .await?;
            assert_eq!(post_scrub.len(), 128);
            assert!(post_scrub.iter().all(PageSample::is_all_zero));
            assert_eq!(backend.dirty_page_count(device.device_uuid).await?, 0);

            let cert_id = Uuid::new_v4();
            backend
                .mark_available(device.device_uuid, cert_id, receipt.duration_us)
                .await?;

            let ready = backend.get_device(device.device_uuid).await?;
            assert_eq!(ready.state, DeviceState::Available);
            assert_eq!(ready.last_cert_id, Some(cert_id));
            assert_eq!(ready.last_scrub_duration_us, Some(receipt.duration_us));

            let _reallocated = backend
                .allocate_slice(
                    device.device_uuid,
                    "sha256-tenant-beta-next".to_string(),
                    format!("job-next-{model:?}"),
                )
                .await?;
        }

        Ok(())
    }

    #[tokio::test]
    async fn enforces_fail_closed_state_transitions_and_quarantine() -> Result<(), HalError> {
        let backend = MockAcceleratorBackend::new(MockClusterConfig::default());
        let devices = backend.discover_devices().await?;
        let b200 = devices
            .iter()
            .find(|d| d.model == AcceleratorModel::NvidiaB200Blackwell)
            .ok_or(HalError::UnknownHardwareModel {
                raw: "nvidia_b200_blackwell".to_string(),
            })?;

        let range = backend
            .allocate_slice(
                b200.device_uuid,
                "tenant-hash-1".to_string(),
                "job-1".to_string(),
            )
            .await?;

        assert!(matches!(
            backend
                .allocate_slice(
                    b200.device_uuid,
                    "tenant-hash-2".to_string(),
                    "job-2".to_string()
                )
                .await,
            Err(HalError::InvalidDeviceState { .. })
        ));
        assert!(matches!(
            backend.zeroize_range_async(b200.device_uuid, range).await,
            Err(HalError::InvalidDeviceState { .. })
        ));

        let dirty_range = backend.release_to_dirty(b200.device_uuid).await?;

        assert!(matches!(
            backend
                .allocate_slice(
                    b200.device_uuid,
                    "tenant-hash-2".to_string(),
                    "job-2".to_string()
                )
                .await,
            Err(HalError::InvalidDeviceState { .. })
        ));
        assert!(matches!(
            backend
                .mark_available(b200.device_uuid, Uuid::new_v4(), 500)
                .await,
            Err(HalError::InvalidDeviceState { .. })
        ));

        let _receipt = backend
            .zeroize_range_async(b200.device_uuid, dirty_range)
            .await?;

        assert!(matches!(
            backend
                .allocate_slice(
                    b200.device_uuid,
                    "tenant-hash-2".to_string(),
                    "job-2".to_string()
                )
                .await,
            Err(HalError::InvalidDeviceState { .. })
        ));

        let mapped_pages = backend
            .mapped_verification_pages(b200.device_uuid, dirty_range)
            .await?
            .ok_or(HalError::InvalidMemoryRange {
                start_addr: dirty_range.start_addr,
                byte_len: dirty_range.byte_len,
                page_size: dirty_range.page_size_bytes,
            })?;
        let target_page = mapped_pages
            .first()
            .copied()
            .ok_or(HalError::PageAddressOutOfBounds {
                device_uuid: b200.device_uuid,
                page_addr: dirty_range.start_addr,
            })?;

        backend
            .inject_residual_byte_for_test(b200.device_uuid, target_page, 512, 0xFF)
            .await?;

        let readback = backend
            .read_back_pages(b200.device_uuid, &[target_page])
            .await?;
        let sample = readback
            .first()
            .ok_or(HalError::PageAddressOutOfBounds {
                device_uuid: b200.device_uuid,
                page_addr: target_page,
            })?;
        assert!(!sample.is_all_zero());
        assert_eq!(sample.non_zero_byte_count(), 1);

        backend
            .quarantine_device(
                b200.device_uuid,
                "residual non-zero byte detected on read-back".to_string(),
            )
            .await?;
        assert!(matches!(
            backend.get_device(b200.device_uuid).await?.state,
            DeviceState::Quarantined { .. }
        ));

        assert!(matches!(
            backend
                .allocate_slice(
                    b200.device_uuid,
                    "tenant-hash-3".to_string(),
                    "job-3".to_string()
                )
                .await,
            Err(HalError::InvalidDeviceState { .. })
        ));
        assert!(matches!(
            backend
                .zeroize_range_async(b200.device_uuid, dirty_range)
                .await,
            Err(HalError::InvalidDeviceState { .. })
        ));
        assert!(matches!(
            backend
                .mark_available(b200.device_uuid, Uuid::new_v4(), 250)
                .await,
            Err(HalError::InvalidDeviceState { .. })
        ));

        Ok(())
    }

    #[tokio::test]
    async fn validates_out_of_bounds_pages_and_unknown_devices() -> Result<(), HalError> {
        let backend = MockAcceleratorBackend::new(MockClusterConfig::default());
        let unknown_uuid = Uuid::new_v4();
        assert!(matches!(
            backend.get_device(unknown_uuid).await,
            Err(HalError::DeviceNotFound { .. })
        ));

        let devices = backend.discover_devices().await?;
        let first_device = devices.first().ok_or(HalError::DeviceNotFound {
            device_uuid: unknown_uuid,
        })?;

        let range = backend
            .allocate_slice(
                first_device.device_uuid,
                "tenant-bounds".to_string(),
                "job-bounds".to_string(),
            )
            .await?;
        let _ = backend.release_to_dirty(first_device.device_uuid).await?;

        let past_end = range.start_addr.saturating_add(range.byte_len);
        assert!(matches!(
            backend
                .read_back_pages(first_device.device_uuid, &[past_end])
                .await,
            Err(HalError::PageAddressOutOfBounds { .. })
        ));
        assert!(matches!(
            backend
                .read_back_pages(first_device.device_uuid, &[range.start_addr + 1])
                .await,
            Err(HalError::PageAddressOutOfBounds { .. })
        ));

        let bad_range = MemoryRange::new(past_end, u64::from(VRAM_PAGE_SIZE_BYTES))?;
        assert!(matches!(
            backend
                .zeroize_range_async(first_device.device_uuid, bad_range)
                .await,
            Err(HalError::InvalidMemoryRange { .. })
        ));
        assert!(matches!(
            backend.get_device(first_device.device_uuid).await?.state,
            DeviceState::DirtyPendingScrub { .. }
        ));

        Ok(())
    }
}
