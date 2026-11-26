#![allow(unsafe_code)]

use std::collections::BTreeMap;
use std::ffi::{c_char, c_void, CStr};
use std::fmt;
use std::ptr;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use chrono::Utc;
use libloading::Library;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::backend::AcceleratorDevice;
use crate::error::HalError;
use crate::model::{
    AcceleratorVendor, BackendKind, DeviceDescriptor, DeviceState, MemoryRange, PageSample,
    ZeroizeReceipt, VRAM_PAGE_SIZE_BYTES,
};
use crate::native::sweep_plan::{
    align_down_4k, compute_sweep_target_bytes, deterministic_device_uuid, infer_model_from_name,
    select_chunk_verification_targets, DEFAULT_SWEEP_RATIO_PCT, DEFAULT_VERIFY_SAMPLE_PAGES,
    SWEEP_CHUNK_LADDER_BYTES,
};

const ROCM_BACKEND_NAME: &str = "rocm-native";
const HIP_SUCCESS: i32 = 0;
const HIP_ERROR_OUT_OF_MEMORY: i32 = 2;
const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;

#[cfg(target_os = "windows")]
const ROCM_DRIVER_CANDIDATES: &[&str] = &[
    "amdhip64.dll",
    "C:\\Program Files\\AMD\\ROCm\\bin\\amdhip64.dll",
];

#[cfg(not(target_os = "windows"))]
const ROCM_DRIVER_CANDIDATES: &[&str] = &[
    "libamdhip64.so",
    "libamdhip64.so.6",
    "libamdhip64.so.5",
    "/opt/rocm/lib/libamdhip64.so",
    "/usr/lib/x86_64-linux-gnu/libamdhip64.so",
];

type HipError = i32;
type HipStream = *mut c_void;

type FnHipGetDeviceCount = unsafe extern "system" fn(*mut i32) -> HipError;
type FnHipSetDevice = unsafe extern "system" fn(i32) -> HipError;
type FnHipDeviceGetPciBusId = unsafe extern "system" fn(*mut c_char, i32, i32) -> HipError;
type FnHipMemGetInfo = unsafe extern "system" fn(*mut usize, *mut usize) -> HipError;
type FnHipMalloc = unsafe extern "system" fn(*mut *mut c_void, usize) -> HipError;
type FnHipMemsetAsync = unsafe extern "system" fn(*mut c_void, i32, usize, HipStream) -> HipError;
type FnHipDeviceSynchronize = unsafe extern "system" fn() -> HipError;
type FnHipMemcpy = unsafe extern "system" fn(*mut c_void, *const c_void, usize, i32) -> HipError;
type FnHipFree = unsafe extern "system" fn(*mut c_void) -> HipError;

/// Configuration for the dynamic AMD ROCm HIP (`libamdhip64.so` / `amdhip64.dll`) backend.
#[derive(Debug, Clone)]
pub struct RocmDriverConfig {
    pub driver_library_path: Option<String>,
    pub sweep_free_vram_pct: u8,
    pub verify_sample_pages: usize,
}

impl Default for RocmDriverConfig {
    fn default() -> Self {
        Self {
            driver_library_path: None,
            sweep_free_vram_pct: DEFAULT_SWEEP_RATIO_PCT,
            verify_sample_pages: DEFAULT_VERIFY_SAMPLE_PAGES,
        }
    }
}

struct HipDriver {
    driver_path: String,
    hip_get_device_count: FnHipGetDeviceCount,
    hip_set_device: FnHipSetDevice,
    hip_device_get_pci_bus_id: Option<FnHipDeviceGetPciBusId>,
    hip_mem_get_info: FnHipMemGetInfo,
    hip_malloc: FnHipMalloc,
    hip_memset_async: FnHipMemsetAsync,
    hip_device_synchronize: FnHipDeviceSynchronize,
    hip_memcpy: FnHipMemcpy,
    hip_free: FnHipFree,
    _lib: Library,
}

impl fmt::Debug for HipDriver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HipDriver")
            .field("driver_path", &self.driver_path)
            .finish()
    }
}

fn hip_check(symbol: &'static str, code: HipError) -> Result<(), HalError> {
    if code == HIP_SUCCESS {
        Ok(())
    } else {
        Err(HalError::NativeDriverFailure {
            backend: ROCM_BACKEND_NAME,
            message: format!("{symbol} returned hipError_t code {code}"),
        })
    }
}

fn load_symbol<T: Copy>(lib: &Library, name: &'static [u8]) -> Result<T, HalError> {
    // SAFETY: Caller supplies the exact AMD HIP C runtime signature for `name`.
    let sym = unsafe { lib.get::<T>(name) }.map_err(|err| HalError::NativeDriverFailure {
        backend: ROCM_BACKEND_NAME,
        message: format!(
            "failed to resolve symbol `{}`: {err}",
            String::from_utf8_lossy(name).trim_end_matches('\0')
        ),
    })?;
    Ok(*sym)
}

impl HipDriver {
    fn load(explicit_path: Option<&str>) -> Result<Self, HalError> {
        let mut attempts: Vec<String> = Vec::new();
        let mut loaded: Option<(String, Library)> = None;

        if let Some(path) = explicit_path {
            // SAFETY: Dynamically loading the configured ROCm HIP library at runtime.
            match unsafe { Library::new(path) } {
                Ok(lib) => loaded = Some((path.to_string(), lib)),
                Err(err) => attempts.push(format!("{path}: {err}")),
            }
        } else {
            for &candidate in ROCM_DRIVER_CANDIDATES {
                // SAFETY: Dynamically probing standard OS ROCm HIP library paths.
                match unsafe { Library::new(candidate) } {
                    Ok(lib) => {
                        loaded = Some((candidate.to_string(), lib));
                        break;
                    }
                    Err(err) => attempts.push(format!("{candidate}: {err}")),
                }
            }
        }

        let Some((driver_path, lib)) = loaded else {
            return Err(HalError::NativeDriverFailure {
                backend: ROCM_BACKEND_NAME,
                message: format!(
                    "could not dynamically load AMD ROCm HIP runtime ({})",
                    attempts.join("; ")
                ),
            });
        };

        let hip_get_device_count: FnHipGetDeviceCount = load_symbol(&lib, b"hipGetDeviceCount\0")?;
        let hip_set_device: FnHipSetDevice = load_symbol(&lib, b"hipSetDevice\0")?;
        let hip_device_get_pci_bus_id: Option<FnHipDeviceGetPciBusId> =
            load_symbol(&lib, b"hipDeviceGetPCIBusId\0").ok();
        let hip_mem_get_info: FnHipMemGetInfo = load_symbol(&lib, b"hipMemGetInfo\0")?;
        let hip_malloc: FnHipMalloc = load_symbol(&lib, b"hipMalloc\0")?;
        let hip_memset_async: FnHipMemsetAsync = load_symbol(&lib, b"hipMemsetAsync\0")?;
        let hip_device_synchronize: FnHipDeviceSynchronize =
            load_symbol(&lib, b"hipDeviceSynchronize\0")?;
        let hip_memcpy: FnHipMemcpy = load_symbol(&lib, b"hipMemcpy\0")?;
        let hip_free: FnHipFree = load_symbol(&lib, b"hipFree\0")?;

        Ok(Self {
            driver_path,
            hip_get_device_count,
            hip_set_device,
            hip_device_get_pci_bus_id,
            hip_mem_get_info,
            hip_malloc,
            hip_memset_async,
            hip_device_synchronize,
            hip_memcpy,
            hip_free,
            _lib: lib,
        })
    }

    fn enumerate_devices(&self) -> Result<Vec<(i32, DeviceDescriptor)>, HalError> {
        let mut count: i32 = 0;
        // SAFETY: Valid pointer to `i32`.
        hip_check("hipGetDeviceCount", unsafe { (self.hip_get_device_count)(&mut count) })?;

        let device_count = usize::try_from(count).unwrap_or(0);
        let mut result = Vec::with_capacity(device_count);

        for ordinal in 0..device_count {
            let dev_id = i32::try_from(ordinal).unwrap_or(0);
            // SAFETY: Valid device ordinal.
            hip_check("hipSetDevice", unsafe { (self.hip_set_device)(dev_id) })?;

            let mut free_bytes: usize = 0;
            let mut total_bytes: usize = 0;
            // SAFETY: Active HIP device context on current thread.
            hip_check("hipMemGetInfo", unsafe {
                (self.hip_mem_get_info)(&mut free_bytes, &mut total_bytes)
            })?;

            let pci_bus_id = if let Some(get_pci) = self.hip_device_get_pci_bus_id {
                let mut pci_buf = [0 as c_char; 64];
                // SAFETY: Valid 64-byte buffer.
                let rc = unsafe { (get_pci)(pci_buf.as_mut_ptr(), 63, dev_id) };
                if rc == HIP_SUCCESS {
                    if let Some(last) = pci_buf.last_mut() {
                        *last = 0;
                    }
                    // SAFETY: Guaranteed NUL-terminated.
                    unsafe { CStr::from_ptr(pci_buf.as_ptr()) }
                        .to_string_lossy()
                        .trim()
                        .to_string()
                } else {
                    format!("rocm:{ordinal}")
                }
            } else {
                format!("rocm:{ordinal}")
            };

            let device_uuid = deterministic_device_uuid(0x02, ordinal, &pci_bus_id);
            let label = "AMD Instinct ROCm GPU".to_string();
            let (model, tpp_rating) = infer_model_from_name(&label, AcceleratorVendor::Amd);

            result.push((
                dev_id,
                DeviceDescriptor {
                    device_uuid,
                    node_name: format!("rocm-host-gpu-{ordinal:02}"),
                    pci_bus_id,
                    vendor: AcceleratorVendor::Amd,
                    model,
                    vendor_model_label: label,
                    sriov_slice_index: None,
                    sriov_parent_uuid: None,
                    tpp_rating,
                    vram_total_bytes: u64::try_from(total_bytes).unwrap_or(u64::MAX),
                    state: DeviceState::Available,
                    last_cert_id: None,
                    last_scrub_duration_us: None,
                },
            ));
        }

        Ok(result)
    }

    fn sweep_and_capture_samples(
        &self,
        device_uuid: Uuid,
        hip_dev: i32,
        sweep_ratio_pct: u8,
        verify_sample_pages: usize,
    ) -> Result<(MemoryRange, u64, u64, BTreeMap<u64, PageSample>), HalError> {
        struct HipAllocGuard<'a> {
            driver: &'a HipDriver,
            ptr: *mut c_void,
            len: usize,
        }
        impl Drop for HipAllocGuard<'_> {
            fn drop(&mut self) {
                if !self.ptr.is_null() {
                    // SAFETY: Pointer was allocated via `hipMalloc` on the active HIP device.
                    let _ = unsafe { (self.driver.hip_free)(self.ptr) };
                }
            }
        }

        // SAFETY: Valid HIP device ordinal.
        hip_check("hipSetDevice", unsafe { (self.hip_set_device)(hip_dev) })?;

        let mut free_bytes: usize = 0;
        let mut total_bytes: usize = 0;
        // SAFETY: Valid pointers on active HIP device.
        hip_check("hipMemGetInfo", unsafe {
            (self.hip_mem_get_info)(&mut free_bytes, &mut total_bytes)
        })?;

        let target_bytes =
            compute_sweep_target_bytes(ROCM_BACKEND_NAME, free_bytes, sweep_ratio_pct)?;

        let started = Instant::now();
        let mut held_chunks: Vec<HipAllocGuard<'_>> = Vec::new();
        let mut swept_bytes: usize = 0;

        for &ladder_size in SWEEP_CHUNK_LADDER_BYTES {
            while swept_bytes < target_bytes {
                let remaining = align_down_4k(target_bytes.saturating_sub(swept_bytes));
                let request_len = remaining.min(ladder_size);
                if request_len < (VRAM_PAGE_SIZE_BYTES as usize) {
                    break;
                }

                let mut dev_ptr: *mut c_void = ptr::null_mut();
                // SAFETY: Valid pointer output and 4KB-aligned `request_len`.
                let rc = unsafe { (self.hip_malloc)(&mut dev_ptr, request_len) };
                if rc == HIP_ERROR_OUT_OF_MEMORY {
                    break;
                }
                hip_check("hipMalloc", rc)?;

                // SAFETY: `dev_ptr` owns `request_len` bytes on active HIP device.
                hip_check("hipMemsetAsync", unsafe {
                    (self.hip_memset_async)(dev_ptr, 0, request_len, ptr::null_mut())
                })?;

                held_chunks.push(HipAllocGuard {
                    driver: self,
                    ptr: dev_ptr,
                    len: request_len,
                });
                swept_bytes = swept_bytes.saturating_add(request_len);
            }
        }

        if held_chunks.is_empty() || swept_bytes == 0 {
            return Err(HalError::NativeDriverFailure {
                backend: ROCM_BACKEND_NAME,
                message: format!(
                    "unable to allocate any VRAM sweep chunks on ROCm device {device_uuid} (free={free_bytes}B)"
                ),
            });
        }

        // SAFETY: Synchronizes all queued async HIP memsets before read-back verification.
        hip_check("hipDeviceSynchronize", unsafe {
            (self.hip_device_synchronize)()
        })?;

        let duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);

        let chunk_lengths: Vec<usize> = held_chunks.iter().map(|c| c.len).collect();
        let targets = select_chunk_verification_targets(&chunk_lengths, verify_sample_pages);
        let mut captured = BTreeMap::new();

        for (c_idx, byte_offset) in targets {
            let Some(chunk) = held_chunks.get(c_idx) else {
                continue;
            };
            let base_usize = chunk.ptr as usize;
            let Some(src_usize) = base_usize.checked_add(byte_offset) else {
                continue;
            };
            let src_addr = u64::try_from(src_usize).unwrap_or(0);
            let mut host_page = Box::new([0u8; VRAM_PAGE_SIZE_BYTES as usize]);

            // SAFETY: `src_usize` lies within live `[chunk.ptr, chunk.ptr + chunk.len)` and
            // `host_page` is a valid 4096-byte host buffer.
            hip_check("hipMemcpy", unsafe {
                (self.hip_memcpy)(
                    host_page.as_mut_ptr().cast::<c_void>(),
                    src_usize as *const c_void,
                    VRAM_PAGE_SIZE_BYTES as usize,
                    HIP_MEMCPY_DEVICE_TO_HOST,
                )
            })?;
            captured.insert(
                src_addr,
                PageSample {
                    page_addr: src_addr,
                    data: host_page,
                },
            );
        }

        let first_base = held_chunks
            .first()
            .map(|c| u64::try_from(c.ptr as usize).unwrap_or(u64::from(VRAM_PAGE_SIZE_BYTES)))
            .unwrap_or(u64::from(VRAM_PAGE_SIZE_BYTES));
        let aligned_base = first_base - (first_base % u64::from(VRAM_PAGE_SIZE_BYTES));
        let swept_len_u64 = u64::try_from(swept_bytes).unwrap_or(u64::from(VRAM_PAGE_SIZE_BYTES));
        let range = MemoryRange::new(aligned_base, swept_len_u64)?;
        let pages_zeroed = range.page_count();

        drop(held_chunks);

        Ok((range, duration_us, pages_zeroed, captured))
    }
}

#[derive(Debug)]
struct HipDeviceRecord {
    hip_dev: i32,
    descriptor: DeviceDescriptor,
    last_captured_samples: BTreeMap<u64, PageSample>,
}

/// Real dynamic AMD ROCm HIP (`libamdhip64.so` / `amdhip64.dll`) sanitization backend.
#[derive(Debug)]
pub struct RocmAcceleratorBackend {
    pub config: RocmDriverConfig,
    driver: Arc<HipDriver>,
    devices: Arc<RwLock<BTreeMap<Uuid, HipDeviceRecord>>>,
    device_count_cached: usize,
}

impl RocmAcceleratorBackend {
    pub fn connect(config: RocmDriverConfig) -> Result<Self, HalError> {
        let driver = Arc::new(HipDriver::load(config.driver_library_path.as_deref())?);
        let enumerated = driver.enumerate_devices()?;
        let device_count_cached = enumerated.len();
        let mut map = BTreeMap::new();
        for (hip_dev, descriptor) in enumerated {
            map.insert(
                descriptor.device_uuid,
                HipDeviceRecord {
                    hip_dev,
                    descriptor,
                    last_captured_samples: BTreeMap::new(),
                },
            );
        }
        Ok(Self {
            config,
            driver,
            devices: Arc::new(RwLock::new(map)),
            device_count_cached,
        })
    }

    pub fn driver_path(&self) -> &str {
        &self.driver.driver_path
    }

    pub fn device_count(&self) -> usize {
        self.device_count_cached
    }
}

#[async_trait]
impl AcceleratorDevice for RocmAcceleratorBackend {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::RocmNative
    }

    async fn discover_devices(&self) -> Result<Vec<DeviceDescriptor>, HalError> {
        let guard = self.devices.read().await;
        let mut list: Vec<DeviceDescriptor> = guard
            .values()
            .map(|r| r.descriptor.clone())
            .collect();
        list.sort_by(|a, b| a.node_name.cmp(&b.node_name));
        Ok(list)
    }

    async fn get_device(&self, device_uuid: Uuid) -> Result<DeviceDescriptor, HalError> {
        let guard = self.devices.read().await;
        guard
            .get(&device_uuid)
            .map(|r| r.descriptor.clone())
            .ok_or(HalError::DeviceNotFound { device_uuid })
    }

    async fn allocate_slice(
        &self,
        device_uuid: Uuid,
        tenant_id_hash: String,
        job_id: String,
    ) -> Result<MemoryRange, HalError> {
        let mut guard = self.devices.write().await;
        let rec = guard
            .get_mut(&device_uuid)
            .ok_or(HalError::DeviceNotFound { device_uuid })?;
        if !matches!(rec.descriptor.state, DeviceState::Available) {
            return Err(HalError::InvalidDeviceState {
                device_uuid,
                expected: "available",
                actual: rec.descriptor.state.status_label().to_string(),
            });
        }
        rec.descriptor.state = DeviceState::Allocated {
            allocation_id: Uuid::now_v7(),
            tenant_id_hash,
            job_id,
            allocated_at: Utc::now(),
        };
        let aligned_len = rec.descriptor.vram_total_bytes
            - (rec.descriptor.vram_total_bytes % u64::from(VRAM_PAGE_SIZE_BYTES));
        MemoryRange::new(u64::from(VRAM_PAGE_SIZE_BYTES), aligned_len.max(u64::from(VRAM_PAGE_SIZE_BYTES)))
    }

    async fn release_to_dirty(&self, device_uuid: Uuid) -> Result<MemoryRange, HalError> {
        let mut guard = self.devices.write().await;
        let rec = guard
            .get_mut(&device_uuid)
            .ok_or(HalError::DeviceNotFound { device_uuid })?;

        let last_tenant_hash = match &rec.descriptor.state {
            DeviceState::Allocated { tenant_id_hash, .. } => tenant_id_hash.clone(),
            DeviceState::Available => "post-job-epilog".to_string(),
            DeviceState::DirtyPendingScrub {
                last_tenant_hash, ..
            } => last_tenant_hash.clone(),
            other => {
                return Err(HalError::InvalidDeviceState {
                    device_uuid,
                    expected: "allocated | available | dirty_pending_scrub",
                    actual: other.status_label().to_string(),
                });
            }
        };

        rec.descriptor.state = DeviceState::DirtyPendingScrub {
            last_tenant_hash,
            released_at: Utc::now(),
            estimated_dirty_ratio_pct: 90,
        };
        let aligned_len = rec.descriptor.vram_total_bytes
            - (rec.descriptor.vram_total_bytes % u64::from(VRAM_PAGE_SIZE_BYTES));
        MemoryRange::new(u64::from(VRAM_PAGE_SIZE_BYTES), aligned_len.max(u64::from(VRAM_PAGE_SIZE_BYTES)))
    }

    async fn zeroize_range_async(
        &self,
        device_uuid: Uuid,
        _range: MemoryRange,
    ) -> Result<ZeroizeReceipt, HalError> {
        let hip_dev = {
            let mut guard = self.devices.write().await;
            let rec = guard
                .get_mut(&device_uuid)
                .ok_or(HalError::DeviceNotFound { device_uuid })?;
            match &rec.descriptor.state {
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
            rec.descriptor.state = DeviceState::Scrubbing {
                started_at: Utc::now(),
            };
            rec.last_captured_samples.clear();
            rec.hip_dev
        };

        let driver = Arc::clone(&self.driver);
        let ratio_pct = self.config.sweep_free_vram_pct;
        let sample_pages = self.config.verify_sample_pages;

        let (swept_range, duration_us, pages_zeroed, captured) = tokio::task::spawn_blocking(
            move || driver.sweep_and_capture_samples(device_uuid, hip_dev, ratio_pct, sample_pages),
        )
        .await
        .map_err(|err| HalError::NativeDriverFailure {
            backend: ROCM_BACKEND_NAME,
            message: format!("ROCm sweep worker thread join error: {err}"),
        })??;

        {
            let mut guard = self.devices.write().await;
            if let Some(rec) = guard.get_mut(&device_uuid) {
                rec.last_captured_samples = captured;
            }
        }

        Ok(ZeroizeReceipt {
            device_uuid,
            range: swept_range,
            duration_us,
            pages_zeroed,
            backend_kind: BackendKind::RocmNative,
        })
    }

    async fn read_back_pages(
        &self,
        device_uuid: Uuid,
        page_addrs: &[u64],
    ) -> Result<Vec<PageSample>, HalError> {
        let guard = self.devices.read().await;
        let rec = guard
            .get(&device_uuid)
            .ok_or(HalError::DeviceNotFound { device_uuid })?;

        let mut out = Vec::with_capacity(page_addrs.len());
        for &addr in page_addrs {
            let sample = rec
                .last_captured_samples
                .get(&addr)
                .cloned()
                .ok_or(HalError::PageAddressOutOfBounds {
                    device_uuid,
                    page_addr: addr,
                })?;
            out.push(sample);
        }
        Ok(out)
    }

    async fn mapped_verification_pages(
        &self,
        device_uuid: Uuid,
        _range: MemoryRange,
    ) -> Result<Option<Vec<u64>>, HalError> {
        let guard = self.devices.read().await;
        let rec = guard
            .get(&device_uuid)
            .ok_or(HalError::DeviceNotFound { device_uuid })?;
        if rec.last_captured_samples.is_empty() {
            Ok(None)
        } else {
            Ok(Some(rec.last_captured_samples.keys().copied().collect()))
        }
    }

    async fn mark_available(
        &self,
        device_uuid: Uuid,
        cert_id: Uuid,
        scrub_duration_us: u64,
    ) -> Result<(), HalError> {
        let mut guard = self.devices.write().await;
        let rec = guard
            .get_mut(&device_uuid)
            .ok_or(HalError::DeviceNotFound { device_uuid })?;
        if !matches!(rec.descriptor.state, DeviceState::Scrubbing { .. }) {
            return Err(HalError::InvalidDeviceState {
                device_uuid,
                expected: "scrubbing",
                actual: rec.descriptor.state.status_label().to_string(),
            });
        }
        rec.descriptor.state = DeviceState::Available;
        rec.descriptor.last_cert_id = Some(cert_id);
        rec.descriptor.last_scrub_duration_us = Some(scrub_duration_us);
        Ok(())
    }

    async fn quarantine_device(
        &self,
        device_uuid: Uuid,
        reason: String,
    ) -> Result<(), HalError> {
        let mut guard = self.devices.write().await;
        let rec = guard
            .get_mut(&device_uuid)
            .ok_or(HalError::DeviceNotFound { device_uuid })?;
        rec.descriptor.state = DeviceState::Quarantined {
            reason,
            quarantined_at: Utc::now(),
        };
        Ok(())
    }
}