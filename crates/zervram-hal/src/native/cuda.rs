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
    compute_chunk_table_sha256, AcceleratorVendor, BackendKind, DeviceDescriptor, DeviceState,
    GpuReductionVerification, MemoryRange, PageSample, VerificationMode, ZeroizeReceipt,
    READBACK_POISON_BYTE, VRAM_PAGE_SIZE_BYTES,
};
use crate::native::nvml;
use crate::native::sweep_plan::{
    align_down_4k, compute_sweep_target_bytes, deterministic_device_uuid, infer_model_from_name,
    select_csprng_chunk_verification_targets, DEFAULT_CANARY_SAMPLE_PAGES,
    DEFAULT_FALLBACK_SAMPLE_PAGES, DEFAULT_SWEEP_RATIO_PCT, SWEEP_CHUNK_LADDER_BYTES,
};

const CUDA_BACKEND_NAME: &str = "cuda-native";
const CUDA_SUCCESS: i32 = 0;
const CUDA_ERROR_OUT_OF_MEMORY: i32 = 2;
const ACC_POISON_U32: u32 = 0xAAAA_AAAA;
const ACC_POISON_U64: u64 = 0xAAAA_AAAA_AAAA_AAAA;

/// Forward-compatible PTX ISA 7.0 (`target sm_50`, 64-bit) kernel that scans 100% of every swept
/// device chunk in coalesced 64-bit grid-stride loads, bitwise-ORs all words into `[p_out_or]`
/// (executing 0 global OR atomics when VRAM is clean), and accumulates the exact count of `u64`
/// words scanned into `[p_out_words]` (initialized with `0xAAAA_AAAA_AAAA_AAAA` poison on device).
/// NUL-terminated as required by `cuModuleLoadData`.
pub(crate) const PTX_VERIFY_ZERO_KERNEL: &[u8] = b"\
.version 7.0\n\
.target sm_50\n\
.address_size 64\n\
\n\
.visible .entry zervram_verify_zero_u64(\n\
    .param .u64 p_buf,\n\
    .param .u64 p_count_u64,\n\
    .param .u64 p_out_or,\n\
    .param .u64 p_out_words\n\
)\n\
{\n\
    .shared .align 8 .u64 s_cnt;\n\
    .reg .pred  %p<5>;\n\
    .reg .b32   %r<6>;\n\
    .reg .b64   %rd<16>;\n\
\n\
    ld.param.u64 %rd0, [p_buf];\n\
    ld.param.u64 %rd1, [p_count_u64];\n\
    ld.param.u64 %rd2, [p_out_or];\n\
    ld.param.u64 %rd3, [p_out_words];\n\
\n\
    mov.u32 %r0, %tid.x;\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mov.u32 %r3, %nctaid.x;\n\
\n\
    mov.u64 %rd6, 0;\n\
    setp.ne.u32 %p0, %r0, 0;\n\
    @%p0 bra SKIP_INIT;\n\
    st.shared.u64 [s_cnt], %rd6;\n\
SKIP_INIT:\n\
    bar.sync 0;\n\
\n\
    mad.lo.u32 %r4, %r1, %r2, %r0;\n\
    mul.lo.u32 %r5, %r3, %r2;\n\
    cvt.u64.u32 %rd4, %r4;\n\
    cvt.u64.u32 %rd5, %r5;\n\
    mov.u64 %rd7, 0;\n\
\n\
LOOP:\n\
    setp.ge.u64 %p1, %rd4, %rd1;\n\
    @%p1 bra DONE_LOOP;\n\
    shl.b64 %rd8, %rd4, 3;\n\
    add.u64 %rd9, %rd0, %rd8;\n\
    ld.global.u64 %rd10, [%rd9];\n\
    or.b64 %rd6, %rd6, %rd10;\n\
    add.u64 %rd7, %rd7, 1;\n\
    add.u64 %rd4, %rd4, %rd5;\n\
    bra LOOP;\n\
\n\
DONE_LOOP:\n\
    setp.eq.u64 %p2, %rd6, 0;\n\
    @%p2 bra SKIP_OR;\n\
    atom.global.or.b64 %rd11, [%rd2], %rd6;\n\
SKIP_OR:\n\
    setp.eq.u64 %p3, %rd7, 0;\n\
    @%p3 bra SKIP_CNT;\n\
    atom.shared.add.u64 %rd12, [s_cnt], %rd7;\n\
SKIP_CNT:\n\
    bar.sync 0;\n\
\n\
    @%p0 bra EXIT;\n\
    ld.shared.u64 %rd13, [s_cnt];\n\
    setp.eq.u64 %p4, %rd13, 0;\n\
    @%p4 bra EXIT;\n\
    atom.global.add.u64 %rd14, [%rd3], %rd13;\n\
EXIT:\n\
    ret;\n\
}\n\0";

const PTX_KERNEL_ENTRY_NAME: &[u8] = b"zervram_verify_zero_u64\0";

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
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
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
type FnCuModuleLoadData = unsafe extern "system" fn(*mut CuModule, *const c_void) -> CuResult;
type FnCuModuleUnload = unsafe extern "system" fn(CuModule) -> CuResult;
type FnCuModuleGetFunction =
    unsafe extern "system" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
type FnCuLaunchKernel = unsafe extern "system" fn(
    CuFunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    CuStream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CuResult;

/// Configuration for the dynamic NVIDIA CUDA Driver API sanitization backend.
#[derive(Debug, Clone)]
pub struct CudaDriverConfig {
    pub driver_library_path: Option<String>,
    pub nvml_library_path: Option<String>,
    /// Percentage of free physical VRAM to sweep per post-job sanitization pass (default: 99%).
    pub sweep_free_vram_pct: u8,
    /// Number of CSPRNG-random 4 KiB canary pages to read back via `cuMemcpyDtoH_v2` alongside
    /// the 100% GPU PTX reduction check (default: 64).
    pub verify_sample_pages: usize,
    /// Number of CSPRNG-random 4 KiB pages to read back if PTX JIT is unavailable (default: 256).
    pub fallback_sample_pages: usize,
    /// When `true`, bypasses the fail-closed NVML active-process safety gate (`--force`).
    pub force_ignore_active_processes: bool,
    /// When `true`, attempts 100% GPU-side PTX reduction verification before falling back.
    pub enable_ptx_full_verification: bool,
}

impl Default for CudaDriverConfig {
    fn default() -> Self {
        Self {
            driver_library_path: None,
            nvml_library_path: None,
            sweep_free_vram_pct: DEFAULT_SWEEP_RATIO_PCT,
            verify_sample_pages: DEFAULT_CANARY_SAMPLE_PAGES,
            fallback_sample_pages: DEFAULT_FALLBACK_SAMPLE_PAGES,
            force_ignore_active_processes: false,
            enable_ptx_full_verification: true,
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
    cu_module_load_data: Option<FnCuModuleLoadData>,
    cu_module_unload: Option<FnCuModuleUnload>,
    cu_module_get_function: Option<FnCuModuleGetFunction>,
    cu_launch_kernel: Option<FnCuLaunchKernel>,
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

struct CudaSweepOutcome {
    range: MemoryRange,
    duration_us: u64,
    pages_zeroed: u64,
    vram_total_bytes: u64,
    vram_free_bytes: u64,
    swept_bytes: u64,
    chunk_count: u32,
    chunk_table_sha256: String,
    active_processes_before: u32,
    verification_mode: VerificationMode,
    gpu_reduction: Option<GpuReductionVerification>,
    captured: BTreeMap<u64, PageSample>,
}

struct CtxGuard<'a> {
    driver: &'a CudaDriver,
    ctx: CuContext,
}

impl Drop for CtxGuard<'_> {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            // SAFETY: Context was created by `cuCtxCreate_v2` on this driver.
            let rc = unsafe { (self.driver.cu_ctx_destroy_v2)(self.ctx) };
            if rc != CUDA_SUCCESS {
                tracing::warn!(cu_result = rc, "cuCtxDestroy_v2 failed during teardown");
            }
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
            let rc = unsafe { (self.driver.cu_mem_free_v2)(self.ptr) };
            if rc != CUDA_SUCCESS {
                tracing::warn!(
                    cu_result = rc,
                    ptr = self.ptr,
                    len = self.len,
                    "cuMemFree_v2 failed during teardown"
                );
            }
        }
    }
}

struct ModuleGuard {
    cu_module_unload: FnCuModuleUnload,
    module: CuModule,
}

impl Drop for ModuleGuard {
    fn drop(&mut self) {
        if !self.module.is_null() {
            // SAFETY: Module was loaded via `cuModuleLoadData` in the active context.
            let rc = unsafe { (self.cu_module_unload)(self.module) };
            if rc != CUDA_SUCCESS {
                tracing::warn!(cu_result = rc, "cuModuleUnload failed during teardown");
            }
        }
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
            load_symbol(&lib, b"cuDeviceGetUuid_v2\0")
                .or_else(|_| load_symbol(&lib, b"cuDeviceGetUuid\0"))
                .ok();
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

        let cu_module_load_data: Option<FnCuModuleLoadData> =
            load_symbol(&lib, b"cuModuleLoadData\0").ok();
        let cu_module_unload: Option<FnCuModuleUnload> =
            load_symbol(&lib, b"cuModuleUnload\0").ok();
        let cu_module_get_function: Option<FnCuModuleGetFunction> =
            load_symbol(&lib, b"cuModuleGetFunction\0").ok();
        let cu_launch_kernel: Option<FnCuLaunchKernel> =
            load_symbol(&lib, b"cuLaunchKernel\0").ok();

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
            cu_module_load_data,
            cu_module_unload,
            cu_module_get_function,
            cu_launch_kernel,
            _lib: lib,
        })
    }

    fn enumerate_devices(&self) -> Result<Vec<(CuDevice, DeviceDescriptor)>, HalError> {
        let mut count: i32 = 0;
        // SAFETY: Valid pointer to `i32` after `cuInit(0)`.
        cu_check("cuDeviceGetCount", unsafe {
            (self.cu_device_get_count)(&mut count)
        })?;

        let device_count = usize::try_from(count).unwrap_or(0);
        let mut result = Vec::with_capacity(device_count);

        for ordinal in 0..device_count {
            let ord_i32 = i32::try_from(ordinal).unwrap_or(0);
            let ord_u32 = u32::try_from(ordinal).unwrap_or(u32::MAX);
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
                device_ordinal: ord_u32,
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

    fn run_ptx_full_reduction(
        &self,
        held_chunks: &[AllocGuard<'_>],
        d_acc_ptr: CuDevicePtr,
        swept_bytes: usize,
    ) -> Result<GpuReductionVerification, HalError> {
        let (Some(load_data), Some(unload), Some(get_func), Some(launch)) = (
            self.cu_module_load_data,
            self.cu_module_unload,
            self.cu_module_get_function,
            self.cu_launch_kernel,
        ) else {
            return Err(HalError::NativeDriverFailure {
                backend: CUDA_BACKEND_NAME,
                message: "CUDA module/kernel launch symbols unavailable in driver".to_string(),
            });
        };

        let mut module: CuModule = ptr::null_mut();
        // SAFETY: `PTX_VERIFY_ZERO_KERNEL` is a NUL-terminated PTX ISA 7.0 string.
        cu_check("cuModuleLoadData", unsafe {
            (load_data)(&mut module, PTX_VERIFY_ZERO_KERNEL.as_ptr().cast::<c_void>())
        })?;
        let _module_guard = ModuleGuard {
            cu_module_unload: unload,
            module,
        };

        let mut func: CuFunction = ptr::null_mut();
        // SAFETY: `PTX_KERNEL_ENTRY_NAME` is NUL-terminated and `module` is loaded in context.
        cu_check("cuModuleGetFunction", unsafe {
            (get_func)(
                &mut func,
                module,
                PTX_KERNEL_ENTRY_NAME.as_ptr().cast::<c_char>(),
            )
        })?;

        let d_out_or = d_acc_ptr;
        let d_out_words = d_acc_ptr
            .checked_add(8)
            .ok_or(HalError::ArithmeticOverflow {
                context: "d_out_words pointer",
            })?;

        // Poison the full 16-byte device accumulator with `0xAA`, then zero only the first 8-byte
        // bitwise-OR slot (`d_out_or`) while leaving `d_out_words == 0xAAAA_AAAA_AAAA_AAAA`.
        // SAFETY: `d_acc_ptr` owns 16 bytes in the active CUDA context.
        cu_check("cuMemsetD32Async(poison acc)", unsafe {
            (self.cu_memset_d32_async)(d_out_or, ACC_POISON_U32, 4, ptr::null_mut())
        })?;
        // SAFETY: `d_out_or` owns the first 8 bytes (2 u32 words) of `d_acc_ptr`.
        cu_check("cuMemsetD32Async(zero or-acc)", unsafe {
            (self.cu_memset_d32_async)(d_out_or, 0u32, 2, ptr::null_mut())
        })?;

        let mut expected_words: u64 = 0;
        for chunk in held_chunks {
            let chunk_words = u64::try_from(chunk.len / 8).unwrap_or(0);
            if chunk_words == 0 {
                continue;
            }
            expected_words = expected_words.saturating_add(chunk_words);

            let block_dim_x: u32 = 256;
            let blocks_needed = chunk_words.div_ceil(u64::from(block_dim_x));
            let grid_dim_x = u32::try_from(blocks_needed.clamp(1, 512)).unwrap_or(512);

            let mut arg_buf: u64 = chunk.ptr;
            let mut arg_count: u64 = chunk_words;
            let mut arg_out_or: u64 = d_out_or;
            let mut arg_out_words: u64 = d_out_words;
            let mut params: [*mut c_void; 4] = [
                (&mut arg_buf as *mut u64).cast::<c_void>(),
                (&mut arg_count as *mut u64).cast::<c_void>(),
                (&mut arg_out_or as *mut u64).cast::<c_void>(),
                (&mut arg_out_words as *mut u64).cast::<c_void>(),
            ];

            // SAFETY: All 4 parameter pointers remain valid during synchronous `cuLaunchKernel`
            // parameter copy; device pointers lie within live allocations in the active context.
            cu_check("cuLaunchKernel(zervram_verify_zero_u64)", unsafe {
                (launch)(
                    func,
                    grid_dim_x,
                    1,
                    1,
                    block_dim_x,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                )
            })?;
        }

        // SAFETY: Active CUDA context on current thread.
        cu_check("cuCtxSynchronize(ptx_verify)", unsafe {
            (self.cu_ctx_synchronize)()
        })?;

        // Pre-poison host accumulator with `0xAA` bytes before D2H copy.
        let mut host_acc: [u64; 2] = [ACC_POISON_U64; 2];
        // SAFETY: `d_out_or` holds 16 readable bytes and `host_acc` holds 16 writable bytes.
        cu_check("cuMemcpyDtoH_v2(ptx_acc)", unsafe {
            (self.cu_memcpy_dtoh_v2)(host_acc.as_mut_ptr().cast::<c_void>(), d_out_or, 16)
        })?;

        let [non_zero_or_mask, raw_words] = host_acc;
        let words_scanned = raw_words.wrapping_sub(ACC_POISON_U64);
        let verified_bytes = u64::try_from(swept_bytes).unwrap_or(u64::MAX);

        Ok(GpuReductionVerification {
            verified_bytes,
            words_scanned,
            expected_words,
            non_zero_or_mask,
        })
    }

    /// Runs the NVML safety gate, sweeps up to `sweep_free_vram_pct` (default 99%) of free VRAM
    /// across the descending chunk ladder until `CUDA_ERROR_OUT_OF_MEMORY`, executes 100% GPU PTX
    /// verification + CSPRNG canary page read-backs into `0xAA`-poisoned host buffers (with
    /// graceful fallback to `csprng_sampled_256_pages` if JIT is unavailable), and frees all
    /// GPU memory + destroys the CUDA context deterministically.
    fn sweep_and_capture_samples(
        &self,
        device_uuid: Uuid,
        pci_bus_id: &str,
        device_ordinal: u32,
        cu_dev: CuDevice,
        config: &CudaDriverConfig,
    ) -> Result<CudaSweepOutcome, HalError> {
        let active_processes_before = match nvml::inspect_device_before_scrub(
            config.nvml_library_path.as_deref(),
            device_uuid,
            pci_bus_id,
            device_ordinal,
        ) {
            Ok(report) => {
                if report.ecc_remapping_failure {
                    return Err(HalError::HardwareFaultQuarantine {
                        device_uuid,
                        reason: "NVML reported uncorrectable ECC row-remapping failure".to_string(),
                    });
                }
                if !report.active_other_pids.is_empty() && !config.force_ignore_active_processes {
                    return Err(HalError::ActiveProcessesDetected {
                        device_uuid,
                        active_pids: report.active_other_pids,
                    });
                }
                u32::try_from(report.active_other_pids.len()).unwrap_or(u32::MAX)
            }
            Err(err) => {
                if !config.force_ignore_active_processes {
                    return Err(HalError::NativeDriverFailure {
                        backend: CUDA_BACKEND_NAME,
                        message: format!(
                            "NVML pre-scrub active-process check failed ({err}); pass --force to bypass"
                        ),
                    });
                }
                tracing::warn!(
                    %device_uuid,
                    error = %err,
                    "--force set: bypassing unavailable NVML active-process check"
                );
                0
            }
        };

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

        // Reserve the tiny 16-byte GPU reduction accumulator BEFORE draining free VRAM to OOM.
        let mut d_acc_ptr: CuDevicePtr = 0;
        // SAFETY: Active CUDA context; allocating 16 bytes for `[out_or, out_words]`.
        let d_acc_guard = if unsafe { (self.cu_mem_alloc_v2)(&mut d_acc_ptr, 16) } == CUDA_SUCCESS
            && d_acc_ptr != 0
        {
            Some(AllocGuard {
                driver: self,
                ptr: d_acc_ptr,
                len: 16,
            })
        } else {
            None
        };

        let target_bytes =
            compute_sweep_target_bytes(CUDA_BACKEND_NAME, free_bytes, config.sweep_free_vram_pct)?;

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
                // SAFETY: Valid active CUDA context; `request_len` is 4 KiB-aligned and non-zero.
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

        // Synchronize all queued async D32 memsets before verification.
        // SAFETY: Active CUDA context on current thread.
        cu_check("cuCtxSynchronize", unsafe { (self.cu_ctx_synchronize)() })?;

        let (verification_mode, gpu_reduction, canary_page_target) =
            if config.enable_ptx_full_verification {
                if let Some(ref acc_guard) = d_acc_guard {
                    match self.run_ptx_full_reduction(&held_chunks, acc_guard.ptr, swept_bytes) {
                        Ok(red) => (
                            VerificationMode::GpuPtxFullReduction,
                            Some(red),
                            config.verify_sample_pages.max(1),
                        ),
                        Err(err) => {
                            let fallback = config.fallback_sample_pages.max(1);
                            let mode = if fallback == DEFAULT_FALLBACK_SAMPLE_PAGES {
                                VerificationMode::CsprngSampled256Pages
                            } else {
                                VerificationMode::CsprngSampledPages
                            };
                            tracing::warn!(
                                %device_uuid,
                                error = %err,
                                fallback_pages = fallback,
                                "CUDA PTX JIT verification unavailable; falling back to CSPRNG host page sampling"
                            );
                            (mode, None, fallback)
                        }
                    }
                } else {
                    let fallback = config.fallback_sample_pages.max(1);
                    let mode = if fallback == DEFAULT_FALLBACK_SAMPLE_PAGES {
                        VerificationMode::CsprngSampled256Pages
                    } else {
                        VerificationMode::CsprngSampledPages
                    };
                    (mode, None, fallback)
                }
            } else {
                let fallback = config.fallback_sample_pages.max(1);
                let mode = if fallback == DEFAULT_FALLBACK_SAMPLE_PAGES {
                    VerificationMode::CsprngSampled256Pages
                } else {
                    VerificationMode::CsprngSampledPages
                };
                (mode, None, fallback)
            };

        // Read back CSPRNG-random canary pages into 0xAA-poisoned host buffers while chunks are live.
        let chunk_lengths: Vec<usize> = held_chunks.iter().map(|c| c.len).collect();
        let targets = select_csprng_chunk_verification_targets(&chunk_lengths, canary_page_target);
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
            // Pre-poison buffer with 0xAA so a no-op D2H copy fails verification.
            let mut host_page = Box::new([READBACK_POISON_BYTE; VRAM_PAGE_SIZE_BYTES as usize]);
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

        if captured.is_empty() {
            return Err(HalError::NativeDriverFailure {
                backend: CUDA_BACKEND_NAME,
                message: format!(
                    "zero canary pages captured during CUDA verification on device {device_uuid}"
                ),
            });
        }

        let duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);

        let chunk_pairs: Vec<(u64, u64)> = held_chunks
            .iter()
            .map(|c| (c.ptr, u64::try_from(c.len).unwrap_or(u64::MAX)))
            .collect();
        let chunk_table_sha256 = compute_chunk_table_sha256(&chunk_pairs);
        let chunk_count = u32::try_from(held_chunks.len()).unwrap_or(u32::MAX);

        let first_base = held_chunks
            .first()
            .map(|c| c.ptr)
            .unwrap_or(u64::from(VRAM_PAGE_SIZE_BYTES));
        let swept_len_u64 =
            u64::try_from(swept_bytes).unwrap_or(u64::from(VRAM_PAGE_SIZE_BYTES));
        let aligned_base = first_base - (first_base % u64::from(VRAM_PAGE_SIZE_BYTES));
        let range = MemoryRange::new(aligned_base, swept_len_u64)?;
        let pages_zeroed = range.page_count();

        // Explicit drop order: free all GPU buffers before destroying CUDA context.
        drop(held_chunks);
        drop(d_acc_guard);
        drop(_ctx_guard);

        Ok(CudaSweepOutcome {
            range,
            duration_us,
            pages_zeroed,
            vram_total_bytes: u64::try_from(total_bytes).unwrap_or(u64::MAX),
            vram_free_bytes: u64::try_from(free_bytes).unwrap_or(u64::MAX),
            swept_bytes: swept_len_u64,
            chunk_count,
            chunk_table_sha256,
            active_processes_before,
            verification_mode,
            gpu_reduction,
            captured,
        })
    }
}

#[derive(Debug)]
struct CudaDeviceRecord {
    cu_device: CuDevice,
    descriptor: DeviceDescriptor,
    last_captured_samples: BTreeMap<u64, PageSample>,
}

/// Dynamic NVIDIA CUDA Driver API (`libcuda.so.1` / `nvcuda.dll`) + NVML sanitization backend.
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
        let mut list: Vec<DeviceDescriptor> =
            guard.values().map(|r| r.descriptor.clone()).collect();
        list.sort_by_key(|d| d.device_ordinal);
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
        MemoryRange::new(
            u64::from(VRAM_PAGE_SIZE_BYTES),
            aligned_len.max(u64::from(VRAM_PAGE_SIZE_BYTES)),
        )
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
            estimated_dirty_ratio_pct: self.config.sweep_free_vram_pct,
        };
        let aligned_len = rec.descriptor.vram_total_bytes
            - (rec.descriptor.vram_total_bytes % u64::from(VRAM_PAGE_SIZE_BYTES));
        MemoryRange::new(
            u64::from(VRAM_PAGE_SIZE_BYTES),
            aligned_len.max(u64::from(VRAM_PAGE_SIZE_BYTES)),
        )
    }

    async fn zeroize_range_async(
        &self,
        device_uuid: Uuid,
        _range: MemoryRange,
    ) -> Result<ZeroizeReceipt, HalError> {
        let (cu_dev, pci_bus_id, device_ordinal) = {
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
            (
                rec.cu_device,
                rec.descriptor.pci_bus_id.clone(),
                rec.descriptor.device_ordinal,
            )
        };

        let driver = Arc::clone(&self.driver);
        let cfg = self.config.clone();

        let outcome_res = tokio::task::spawn_blocking(move || {
            driver.sweep_and_capture_samples(device_uuid, &pci_bus_id, device_ordinal, cu_dev, &cfg)
        })
        .await
        .map_err(|err| HalError::NativeDriverFailure {
            backend: CUDA_BACKEND_NAME,
            message: format!("CUDA sweep worker thread join error: {err}"),
        })?;

        let outcome = match outcome_res {
            Ok(outcome) => outcome,
            Err(HalError::HardwareFaultQuarantine {
                device_uuid: fault_uuid,
                reason,
            }) => {
                let mut guard = self.devices.write().await;
                if let Some(rec) = guard.get_mut(&device_uuid) {
                    rec.descriptor.state = DeviceState::Quarantined {
                        reason: reason.clone(),
                        quarantined_at: Utc::now(),
                    };
                }
                return Err(HalError::HardwareFaultQuarantine {
                    device_uuid: fault_uuid,
                    reason,
                });
            }
            Err(err) => return Err(err),
        };

        {
            let mut guard = self.devices.write().await;
            if let Some(rec) = guard.get_mut(&device_uuid) {
                rec.last_captured_samples = outcome.captured;
            }
        }

        Ok(ZeroizeReceipt {
            device_uuid,
            range: outcome.range,
            duration_us: outcome.duration_us,
            pages_zeroed: outcome.pages_zeroed,
            backend_kind: BackendKind::CudaNative,
            vram_total_bytes: outcome.vram_total_bytes,
            vram_free_bytes: outcome.vram_free_bytes,
            swept_bytes: outcome.swept_bytes,
            chunk_count: outcome.chunk_count,
            chunk_table_sha256: outcome.chunk_table_sha256,
            active_processes_before: outcome.active_processes_before,
            verification_mode: outcome.verification_mode,
            gpu_reduction: outcome.gpu_reduction,
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