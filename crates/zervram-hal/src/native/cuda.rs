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

const CUDA_BACKEND_NAME: &str = "cuda-native";
const CUDA_SUCCESS: i32 = 0;
const CUDA_ERROR_OUT_OF_MEMORY: i32 = 2;

#[cfg(target_os = "windows")]
const CUDA_DRIVER_CANDIDATES: &[&str] = &["nvcuda.dll", "C:\\Windows\\System32\\nvcuda.dll"];

#[cfg(not(target_os = "windows"))]
const CUDA_DRIVER_CANDIDATES: &[&str] = &[
    "libcuda.so.1",
    "libcuda.so",
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/lib64/libcuda.so.1",
    "/usr/local/cuda/lib64/libcuda.so.1",
];

type CuResult = i32;
type CuDevice = i32;
type CuContext = *mut c_void;
type CuStream = *mut c_void;
type CuDevicePtr = u64;

type FnCuInit = unsafe extern "system" fn(u32) -> CuResult;
type FnCuDeviceGetCount = unsafe extern "system" fn(*mut i32) -> CuResult;
type FnCuDeviceGet = unsafe extern "system" fn(*mut CuDevice, i32) -> CuResult;
type FnCuDeviceGetName = unsafe extern "system" fn(*mut c_char, i32, CuDevice) -> CuResult;
type FnCuDeviceGetUuid = unsafe extern "system" fn(*mut [u8; 16], CuDevice) -> CuResult;
type FnCuDeviceGetPciBusId = unsafe extern "system" fn(*mut c_char, i32, CuDevice) -> CuResult;
type FnCuDeviceTotalMemV2 = unsafe extern "system" fn(*mut usize, CuDevice) -> CuResult;
type FnCuCtxCreateV2 = unsafe extern "system" fn(*mut CuContext, u32, CuDevice) -> CuResult;
type FnCuCtxSynchronize = unsafe extern "system" fn() -> CuResult;
type FnCuCtxDestroyV2 = unsafe extern "system" fn(CuContext) -> CuResult;
type FnCuMemGetInfoV2 = unsafe extern "system" fn(*mut usize, *mut usize) -> CuResult;
type FnCuMemAllocV2 = unsafe extern "system" fn(*mut CuDevicePtr, usize) -> CuResult;
type FnCuMemsetD32Async = unsafe extern "system" fn(CuDevicePtr, u32, usize, CuStream) -> CuResult;
type FnCuMemcpyDtoHV2 = unsafe extern "system" fn(*mut c_void, CuDevicePtr, usize) -> CuResult;
type FnCuMemFreeV2 = unsafe extern "system" fn(CuDevicePtr) -> CuResult;

/// Configuration for the dynamic NVIDIA CUDA Driver API sanitization backend.
#[derive(Debug, Clone)]
pub struct CudaDriverConfig {
    pub driver_library_path: Option<String>,
    /// Percentage of free physical VRAM to sweep per post-job sanitization pass (default: 90%).
    pub sweep_free_vram_pct: u8,
    /// Number of 4KB pages to spot-read back via `cuMemcpyDtoH_v2` before freeing chunks (default: 64).
    pub verify_sample_pages: usize,
}

impl Default for CudaDriverConfig {
    fn default() -> Self {
        Self {
            driver_library_path: None,
            sweep_free_vram_pct: DEFAULT_SWEEP_RATIO_PCT,
            verify_sample_pages: DEFAULT_VERIFY_SAMPLE_PAGES,
        }
    }
}

struct CudaDriver {
    driver_path: String,
    cu_init: FnCuInit,
    cu_device_get_count: FnCuDeviceGetCount,
    cu_device_get: FnCuDeviceGet,
    cu_device_get_name: FnCuDeviceGetName,
    cu_device_get_uuid: Option<FnCuDeviceGetUuid>,
    cu_device_get_pci_bus_id: Option<FnCuDeviceGetPciBusId>,
    cu_device_total_mem_v2: FnCuDeviceTotalMemV2,
    cu_ctx_create_v2: FnCuCtxCreateV2,
    cu_ctx_synchronize: FnCuCtxSynchronize,
    cu_ctx_destroy_v2: FnCuCtxDestroyV2,
    cu_mem_get_info_v2: FnCuMemGetInfoV2,
    cu_mem_alloc_v2: FnCuMemAllocV2,
    cu_memset_d32_async: FnCuMemsetD32Async,
    cu_memcpy_dtoh_v2: FnCuMemcpyDtoHV2,
    cu_mem_free_v2: FnCuMemFreeV2,
    // Kept alive inside this struct for the lifetime of all resolved function pointers.
    _lib: Library,
}

impl fmt::Debug for CudaDriver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaDriver")
            .field("driver_path", &self.driver_path)
            .finish()
    }
}

fn cu_check(symbol: &'static str, code: CuResult) -> Result<(), HalError> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(HalError::NativeDriverFailure {
            backend: CUDA_BACKEND_NAME,
            message: format!("{symbol} returned CUresult error code {code}"),
        })
    }
}

fn load_symbol<T: Copy>(lib: &Library, name: &'static [u8]) -> Result<T, HalError> {
    // SAFETY: Caller supplies the exact NVIDIA CUDA Driver API C signature for `name`.
    let sym = unsafe { lib.get::<T>(name) }.map_err(|err| HalError::NativeDriverFailure {
        backend: CUDA_BACKEND_NAME,
        message: format!(
            "failed to resolve symbol `{}`: {err}",
            String::from_utf8_lossy(name).trim_end_matches('\0')
        ),
    })?;
    Ok(*sym)
}

impl CudaDriver {
    fn load(explicit_path: Option<&str>) -> Result<Self, HalError> {
        let mut attempts: Vec<String> = Vec::new();
        let mut loaded: Option<(String, Library)> = None;

        if let Some(path) = explicit_path {
            // SAFETY: Dynamically loading the configured CUDA driver library at runtime.
            match unsafe { Library::new(path) } {
                Ok(lib) => loaded = Some((path.to_string(), lib)),
                Err(err) => attempts.push(format!("{path}: {err}")),
            }
        } else {
            for &candidate in CUDA_DRIVER_CANDIDATES {
                // SAFETY: Dynamically probing standard OS CUDA driver library paths.
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
                backend: CUDA_BACKEND_NAME,
                message: format!(
                    "could not dynamically load NVIDIA CUDA driver ({})",
                    attempts.join("; ")
                ),
            });
        };

        let cu_init: FnCuInit = load_symbol(&lib, b"cuInit\0")?;
        let cu_device_get_count: FnCuDeviceGetCount = load_symbol(&lib, b"cuDeviceGetCount\0")?;
        let cu_device_get: FnCuDeviceGet = load_symbol(&lib, b"cuDeviceGet\0")?;
        let cu_device_get_name: FnCuDeviceGetName = load_symbol(&lib, b"cuDeviceGetName\0")?;
        let cu_device_get_uuid: Option<FnCuDeviceGetUuid> =
            load_symbol(&lib, b"cuDeviceGetUuid\0").ok();
        let cu_device_get_pci_bus_id: Option<FnCuDeviceGetPciBusId> =
            load_symbol(&lib, b"cuDeviceGetPCIBusId\0").ok();
        let cu_device_total_mem_v2: FnCuDeviceTotalMemV2 =
            load_symbol(&lib, b"cuDeviceTotalMem_v2\0")?;
        let cu_ctx_create_v2: FnCuCtxCreateV2 = load_symbol(&lib, b"cuCtxCreate_v2\0")?;
        let cu_ctx_synchronize: FnCuCtxSynchronize = load_symbol(&lib, b"cuCtxSynchronize\0")?;
        let cu_ctx_destroy_v2: FnCuCtxDestroyV2 = load_symbol(&lib, b"cuCtxDestroy_v2\0")?;
        let cu_mem_get_info_v2: FnCuMemGetInfoV2 = load_symbol(&lib, b"cuMemGetInfo_v2\0")?;
        let cu_mem_alloc_v2: FnCuMemAllocV2 = load_symbol(&lib, b"cuMemAlloc_v2\0")?;
        let cu_memset_d32_async: FnCuMemsetD32Async = load_symbol(&lib, b"cuMemsetD32Async\0")?;
        let cu_memcpy_dtoh_v2: FnCuMemcpyDtoHV2 = load_symbol(&lib, b"cuMemcpyDtoH_v2\0")?;
        let cu_mem_free_v2: FnCuMemFreeV2 = load_symbol(&lib, b"cuMemFree_v2\0")?;

        // SAFETY: `cuInit(0)` initializes the CUDA Driver API once per process.
        cu_check("cuInit", unsafe { (cu_init)(0) })?;

        Ok(Self {
            driver_path,
            cu_init,
            cu_device_get_count,
            cu_device_get,
            cu_device_get_name,
            cu_device_get_uuid,
            cu_device_get_pci_bus_id,
            cu_device_total_mem_v2,
            cu_ctx_create_v2,
            cu_ctx_synchronize,
            cu_ctx_destroy_v2,
            cu_mem_get_info_v2,
            cu_mem_alloc_v2,
            cu_memset_d32_async,
            cu_memcpy_dtoh_v2,
            cu_mem_free_v2,
            _lib: lib,
        })
    }

    fn enumerate_devices(&self) -> Result<Vec<(CuDevice, DeviceDescriptor)>, HalError> {
        let mut count: i32 = 0;
        // SAFETY: Valid pointer to `i32` after `cuInit(0)`.
        cu_check("cuDeviceGetCount", unsafe { (self.cu_device_get_count)(&mut count) })?;

        let device_count = usize::try_from(count).unwrap_or(0);
        let mut result = Vec::with_capacity(device_count);

        for ordinal in 0..device_count {
            let ord_i32 = i32::try_from(ordinal).unwrap_or(0);
            let mut dev: CuDevice = 0;
            // SAFETY: Valid output pointer and in-bounds device ordinal.
            cu_check("cuDeviceGet", unsafe { (self.cu_device_get)(&mut dev, ord_i32) })?;

            let mut name_buf = [0 as c_char; 256];
            // SAFETY: Buffer length matches `name_buf.len()` and we force NUL-termination.
            cu_check("cuDeviceGetName", unsafe {
                (self.cu_device_get_name)(name_buf.as_mut_ptr(), 255, dev)
            })?;
            if let Some(last) = name_buf.last_mut() {
                *last = 0;
            }
            // SAFETY: `name_buf` is guaranteed NUL-terminated.
            let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
                .to_string_lossy()
                .trim()
                .to_string();

            let pci_bus_id = if let Some(get_pci) = self.cu_device_get_pci_bus_id {
                let mut pci_buf = [0 as c_char; 64];
                // SAFETY: Valid buffer of 64 bytes.
                let rc = unsafe { (get_pci)(pci_buf.as_mut_ptr(), 63, dev) };
                if rc == CUDA_SUCCESS {
                    if let Some(last) = pci_buf.last_mut() {
                        *last = 0;
                    }
                    // SAFETY: Guaranteed NUL-terminated.
                    unsafe { CStr::from_ptr(pci_buf.as_ptr()) }
                        .to_string_lossy()
                        .trim()
                        .to_string()
                } else {
                    format!("cuda:{ordinal}")
                }
            } else {
                format!("cuda:{ordinal}")
            };

            let device_uuid = if let Some(get_uuid) = self.cu_device_get_uuid {
                let mut raw = [0u8; 16];
                // SAFETY: Valid 16-byte buffer.
                if unsafe { (get_uuid)(&mut raw, dev) } == CUDA_SUCCESS {
                    Uuid::from_bytes(raw)
                } else {
                    deterministic_device_uuid(0x01, ordinal, &pci_bus_id)
                }
            } else {
                deterministic_device_uuid(0x01, ordinal, &pci_bus_id)
            };

            let mut total_mem: usize = 0;
            // SAFETY: Valid pointer to `usize`.
            cu_check("cuDeviceTotalMem_v2", unsafe {
                (self.cu_device_total_mem_v2)(&mut total_mem, dev)
            })?;
            let vram_total_bytes = u64::try_from(total_mem).unwrap_or(u64::MAX);

            let (model, tpp_rating) = infer_model_from_name(&name, AcceleratorVendor::Nvidia);
            let descriptor = DeviceDescriptor {
                device_uuid,
                node_name: format!("cuda-host-gpu-{ordinal:02}"),
                pci_bus_id,
                vendor: AcceleratorVendor::Nvidia,
                model,
                vendor_model_label: if name.is_empty() {
                    "NVIDIA CUDA GPU".to_string()
                } else {
                    name
                },
                sriov_slice_index: None,
                sriov_parent_uuid: None,
                tpp_rating,
                vram_total_bytes,
                state: DeviceState::Available,
                last_cert_id: None,
                last_scrub_duration_us: None,
            };

            result.push((dev, descriptor));
        }

        Ok(result)
    }

    /// Allocates and sweeps ~90% of free physical VRAM on `cu_dev` in safe descending chunks,
    /// synchronizes once for peak HBM/GDDR bandwidth, spot-reads 4KB pages via `cuMemcpyDtoH_v2`
    /// while chunks are live, and deterministically frees all GPU buffers + destroys the context.
    fn sweep_and_capture_samples(
        &self,
        device_uuid: Uuid,
        cu_dev: CuDevice,
        sweep_ratio_pct: u8,
        verify_sample_pages: usize,
    ) -> Result<(MemoryRange, u64, u64, BTreeMap<u64, PageSample>), HalError> {
        struct CtxGuard<'a> {
            driver: &'a CudaDriver,
            ctx: CuContext,
        }
        impl Drop for CtxGuard<'_> {
            fn drop(&mut self) {
                if !self.ctx.is_null() {
                    // SAFETY: Context was created by `cuCtxCreate_v2` on this driver.
                    let _ = unsafe { (self.driver.cu_ctx_destroy_v2)(self.ctx) };
                }
            }
        }

        struct AllocGuard<'a> {
            driver: &'a CudaDriver,
            ptr: CuDevicePtr,
            len: usize,
        }
        impl Drop for AllocGuard<'_> {
            fn drop(&mut self) {
                if self.ptr != 0 {
                    // SAFETY: Pointer was returned by `cuMemAlloc_v2` in the active context.
                    let _ = unsafe { (self.driver.cu_mem_free_v2)(self.ptr) };
                }
            }
        }

        // Re-assert initialization (no-op if already initialized).
        // SAFETY: `cuInit(0)` is idempotent.
        cu_check("cuInit", unsafe { (self.cu_init)(0) })?;

        let mut ctx: CuContext = ptr::null_mut();
        // SAFETY: Valid context output pointer and CUDA device handle.
        cu_check("cuCtxCreate_v2", unsafe {
            (self.cu_ctx_create_v2)(&mut ctx, 0, cu_dev)
        })?;
        let _ctx_guard = CtxGuard { driver: self, ctx };

        let mut free_bytes: usize = 0;
        let mut total_bytes: usize = 0;
        // SAFETY: Active CUDA context on current OS thread.
        cu_check("cuMemGetInfo_v2", unsafe {
            (self.cu_mem_get_info_v2)(&mut free_bytes, &mut total_bytes)
        })?;

        let target_bytes =
            compute_sweep_target_bytes(CUDA_BACKEND_NAME, free_bytes, sweep_ratio_pct)?;

        let started = Instant::now();
        let mut held_chunks: Vec<AllocGuard<'_>> = Vec::new();
        let mut swept_bytes: usize = 0;

        for &ladder_size in SWEEP_CHUNK_LADDER_BYTES {
            while swept_bytes < target_bytes {
                let remaining = align_down_4k(target_bytes.saturating_sub(swept_bytes));
                let request_len = remaining.min(ladder_size);
                if request_len < (VRAM_PAGE_SIZE_BYTES as usize) {
                    break;
                }

                let mut dptr: CuDevicePtr = 0;
                // SAFETY: Valid active CUDA context; `request_len` is 4KB-aligned and non-zero.
                let rc = unsafe { (self.cu_mem_alloc_v2)(&mut dptr, request_len) };
                if rc == CUDA_ERROR_OUT_OF_MEMORY {
                    break;
                }
                cu_check("cuMemAlloc_v2", rc)?;

                // `cuMemsetD32Async` takes count of 32-bit elements (request_len / 4).
                let count_u32 = request_len / 4;
                // SAFETY: `dptr` is >= 256B aligned and owns `request_len` bytes in active context.
                cu_check("cuMemsetD32Async", unsafe {
                    (self.cu_memset_d32_async)(dptr, 0u32, count_u32, ptr::null_mut())
                })?;

                held_chunks.push(AllocGuard {
                    driver: self,
                    ptr: dptr,
                    len: request_len,
                });
                swept_bytes = swept_bytes.saturating_add(request_len);
            }
        }

        if held_chunks.is_empty() || swept_bytes == 0 {
            return Err(HalError::NativeDriverFailure {
                backend: CUDA_BACKEND_NAME,
                message: format!(
                    "unable to allocate any VRAM sweep chunks on device {device_uuid} (free={free_bytes}B)"
                ),
            });
        }

        // Synchronize all queued async D32 memsets before spot-reading pages back to host RAM.
        // SAFETY: Active CUDA context on current thread.
        cu_check("cuCtxSynchronize", unsafe { (self.cu_ctx_synchronize)() })?;

        let duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);

        // Spot-read 4KB pages via `cuMemcpyDtoH_v2` while all swept device pointers are live.
        let chunk_lengths: Vec<usize> = held_chunks.iter().map(|c| c.len).collect();
        let targets = select_chunk_verification_targets(&chunk_lengths, verify_sample_pages);
        let mut captured = BTreeMap::new();

        for (c_idx, byte_offset) in targets {
            let Some(chunk) = held_chunks.get(c_idx) else {
                continue;
            };
            let Some(src_addr) = chunk
                .ptr
                .checked_add(u64::try_from(byte_offset).unwrap_or(0))
            else {
                continue;
            };
            let mut host_page = Box::new([0u8; VRAM_PAGE_SIZE_BYTES as usize]);
            // SAFETY: `src_addr` lies within `[chunk.ptr, chunk.ptr + chunk.len)` and `host_page`
            // is a valid 4096-byte writable host buffer.
            cu_check("cuMemcpyDtoH_v2", unsafe {
                (self.cu_memcpy_dtoh_v2)(
                    host_page.as_mut_ptr().cast::<c_void>(),
                    src_addr,
                    VRAM_PAGE_SIZE_BYTES as usize,
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
            .map(|c| c.ptr)
            .unwrap_or(u64::from(VRAM_PAGE_SIZE_BYTES));
        let swept_len_u64 = u64::try_from(swept_bytes).unwrap_or(u64::from(VRAM_PAGE_SIZE_BYTES));
        let aligned_base = first_base - (first_base % u64::from(VRAM_PAGE_SIZE_BYTES));
        let range = MemoryRange::new(aligned_base, swept_len_u64)?;
        let pages_zeroed = range.page_count();

        // Explicit drop order: free all GPU buffers before destroying CUDA context.
        drop(held_chunks);
        drop(_ctx_guard);

        Ok((range, duration_us, pages_zeroed, captured))
    }
}

#[derive(Debug)]
struct CudaDeviceRecord {
    cu_device: CuDevice,
    descriptor: DeviceDescriptor,
    last_captured_samples: BTreeMap<u64, PageSample>,
}

/// Real dynamic NVIDIA CUDA Driver API (`libcuda.so.1` / `nvcuda.dll`) sanitization backend.
#[derive(Debug)]
pub struct CudaAcceleratorBackend {
    pub config: CudaDriverConfig,
    driver: Arc<CudaDriver>,
    devices: Arc<RwLock<BTreeMap<Uuid, CudaDeviceRecord>>>,
    device_count_cached: usize,
}

impl CudaAcceleratorBackend {
    pub fn connect(config: CudaDriverConfig) -> Result<Self, HalError> {
        let driver = Arc::new(CudaDriver::load(config.driver_library_path.as_deref())?);
        let enumerated = driver.enumerate_devices()?;
        let device_count_cached = enumerated.len();
        let mut map = BTreeMap::new();
        for (cu_device, descriptor) in enumerated {
            map.insert(
                descriptor.device_uuid,
                CudaDeviceRecord {
                    cu_device,
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
impl AcceleratorDevice for CudaAcceleratorBackend {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::CudaNative
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
        let cu_dev = {
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
            rec.cu_device
        };

        let driver = Arc::clone(&self.driver);
        let ratio_pct = self.config.sweep_free_vram_pct;
        let sample_pages = self.config.verify_sample_pages;

        let (swept_range, duration_us, pages_zeroed, captured) = tokio::task::spawn_blocking(
            move || driver.sweep_and_capture_samples(device_uuid, cu_dev, ratio_pct, sample_pages),
        )
        .await
        .map_err(|err| HalError::NativeDriverFailure {
            backend: CUDA_BACKEND_NAME,
            message: format!("CUDA sweep worker thread join error: {err}"),
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
            backend_kind: BackendKind::CudaNative,
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