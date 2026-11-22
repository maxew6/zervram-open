use crate::error::HalError;
use crate::model::{AcceleratorModel, AcceleratorVendor, HardwareSpec, VRAM_PAGE_SIZE_BYTES};

pub const AGGREGATE_TPP_STATUTORY_CEILING: u64 = 12_000_000;

/// Statutory formula (15 CFR EAR § 744.24): `TPP = 2 * Mac_TOPS * bit_length`.
pub fn calculate_tpp(mac_tops: f64, bit_length: u16) -> Result<u64, HalError> {
    if !mac_tops.is_finite() || mac_tops <= 0.0 || bit_length == 0 {
        return Err(HalError::InvalidTppParameters {
            mac_tops,
            bit_length,
        });
    }
    let raw = 2.0 * mac_tops * f64::from(bit_length);
    if !raw.is_finite() || raw < 0.0 || raw >= (u64::MAX as f64) {
        return Err(HalError::ArithmeticOverflow {
            context: "calculate_tpp",
        });
    }
    Ok(raw.round() as u64)
}

pub fn hardware_spec(model: AcceleratorModel) -> &'static HardwareSpec {
    match model {
        AcceleratorModel::NvidiaH100Sxm => &SPEC_H100_SXM,
        AcceleratorModel::NvidiaB200Blackwell => &SPEC_B200_BLACKWELL,
        AcceleratorModel::AmdInstinctMi300x => &SPEC_MI300X,
        AcceleratorModel::TenstorrentBlackholeP150 => &SPEC_BLACKHOLE_P150,
        AcceleratorModel::FuriosaRngd => &SPEC_FURIOSA_RNGD,
        AcceleratorModel::FuriosaRngdSriovSlice => &SPEC_FURIOSA_RNGD_SLICE,
    }
}

pub fn all_hardware_specs() -> &'static [&'static HardwareSpec] {
    static ALL_SPECS: [&HardwareSpec; 6] = [
        &SPEC_H100_SXM,
        &SPEC_B200_BLACKWELL,
        &SPEC_MI300X,
        &SPEC_BLACKHOLE_P150,
        &SPEC_FURIOSA_RNGD,
        &SPEC_FURIOSA_RNGD_SLICE,
    ];
    &ALL_SPECS
}

/// Checked aggregate TPP multiplication for `quantity` units of `model`.
pub fn total_tpp_for_units(model: AcceleratorModel, quantity: u32) -> Result<u64, HalError> {
    let spec = hardware_spec(model);
    spec.tpp_per_unit
        .checked_mul(u64::from(quantity))
        .ok_or(HalError::ArithmeticOverflow {
            context: "total_tpp_for_units",
        })
}

pub fn resolve_model_from_k8s_resource(
    resource_key: &str,
    model_hint: Option<&str>,
) -> Result<AcceleratorModel, HalError> {
    match resource_key.trim() {
        "nvidia.com/h100" | "nvidia.com/h100-sxm" => Ok(AcceleratorModel::NvidiaH100Sxm),
        "nvidia.com/b200" | "nvidia.com/b200-blackwell" => {
            Ok(AcceleratorModel::NvidiaB200Blackwell)
        }
        "nvidia.com/gpu" => {
            let normalized = model_hint.map(|s| s.trim().to_ascii_lowercase());
            match normalized.as_deref() {
                Some("b200") | Some("nvidia-b200") | Some("b200-blackwell") => {
                    Ok(AcceleratorModel::NvidiaB200Blackwell)
                }
                Some("h100") | Some("nvidia-h100") | Some("h100-sxm") | None => {
                    Ok(AcceleratorModel::NvidiaH100Sxm)
                }
                Some(other) => Err(HalError::UnknownHardwareModel {
                    raw: other.to_string(),
                }),
            }
        }
        "amd.com/mi300x" | "amd.com/gpu" => Ok(AcceleratorModel::AmdInstinctMi300x),
        "tenstorrent.com/blackhole-p150" | "tenstorrent.com/blackhole" => {
            Ok(AcceleratorModel::TenstorrentBlackholeP150)
        }
        "furiosa.ai/rngd" => Ok(AcceleratorModel::FuriosaRngd),
        "furiosa.ai/rngd-sriov" | "furiosa.ai/rngd-slice" => {
            Ok(AcceleratorModel::FuriosaRngdSriovSlice)
        }
        other => Err(HalError::UnknownHardwareModel {
            raw: other.to_string(),
        }),
    }
}

const GIB: u64 = 1024 * 1024 * 1024;

pub const SPEC_H100_SXM: HardwareSpec = HardwareSpec {
    model: AcceleratorModel::NvidiaH100Sxm,
    vendor: AcceleratorVendor::Nvidia,
    canonical_name: "NVIDIA H100 SXM",
    tpp_per_unit: 15_832,
    mac_tops: 989.5,
    bit_length: 8,
    vram_total_bytes: 80 * GIB,
    page_size_bytes: VRAM_PAGE_SIZE_BYTES,
    max_units_under_12m_tpp: 757,
    sriov_slice_count: 1,
    k8s_resource_keys: &["nvidia.com/h100", "nvidia.com/gpu"],
};

pub const SPEC_B200_BLACKWELL: HardwareSpec = HardwareSpec {
    model: AcceleratorModel::NvidiaB200Blackwell,
    vendor: AcceleratorVendor::Nvidia,
    canonical_name: "NVIDIA B200 Blackwell",
    tpp_per_unit: 36_000,
    mac_tops: 2_250.0,
    bit_length: 8,
    vram_total_bytes: 192 * GIB,
    page_size_bytes: VRAM_PAGE_SIZE_BYTES,
    max_units_under_12m_tpp: 333,
    sriov_slice_count: 1,
    k8s_resource_keys: &["nvidia.com/b200", "nvidia.com/gpu"],
};

pub const SPEC_MI300X: HardwareSpec = HardwareSpec {
    model: AcceleratorModel::AmdInstinctMi300x,
    vendor: AcceleratorVendor::Amd,
    canonical_name: "AMD Instinct MI300X",
    tpp_per_unit: 20_880,
    mac_tops: 1_305.0,
    bit_length: 8,
    vram_total_bytes: 192 * GIB,
    page_size_bytes: VRAM_PAGE_SIZE_BYTES,
    max_units_under_12m_tpp: 574,
    sriov_slice_count: 1,
    k8s_resource_keys: &["amd.com/mi300x", "amd.com/gpu"],
};

pub const SPEC_BLACKHOLE_P150: HardwareSpec = HardwareSpec {
    model: AcceleratorModel::TenstorrentBlackholeP150,
    vendor: AcceleratorVendor::Tenstorrent,
    canonical_name: "Tenstorrent Blackhole p150",
    tpp_per_unit: 9_400,
    mac_tops: 587.5,
    bit_length: 8,
    vram_total_bytes: 32 * GIB,
    page_size_bytes: VRAM_PAGE_SIZE_BYTES,
    max_units_under_12m_tpp: 1_276,
    sriov_slice_count: 1,
    k8s_resource_keys: &["tenstorrent.com/blackhole-p150"],
};

pub const SPEC_FURIOSA_RNGD: HardwareSpec = HardwareSpec {
    model: AcceleratorModel::FuriosaRngd,
    vendor: AcceleratorVendor::FuriosaAi,
    canonical_name: "FuriosaAI RNGD",
    tpp_per_unit: 8_192,
    mac_tops: 512.0,
    bit_length: 8,
    vram_total_bytes: 48 * GIB,
    page_size_bytes: VRAM_PAGE_SIZE_BYTES,
    max_units_under_12m_tpp: 1_464,
    sriov_slice_count: 8,
    k8s_resource_keys: &["furiosa.ai/rngd"],
};

pub const SPEC_FURIOSA_RNGD_SLICE: HardwareSpec = HardwareSpec {
    model: AcceleratorModel::FuriosaRngdSriovSlice,
    vendor: AcceleratorVendor::FuriosaAi,
    canonical_name: "FuriosaAI RNGD (SR-IOV 1/8)",
    tpp_per_unit: 1_024,
    mac_tops: 64.0,
    bit_length: 8,
    vram_total_bytes: 6 * GIB,
    page_size_bytes: VRAM_PAGE_SIZE_BYTES,
    max_units_under_12m_tpp: 11_718,
    sriov_slice_count: 1,
    k8s_resource_keys: &["furiosa.ai/rngd-sriov"],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statutory_hardware_specs_match_formula_and_12m_ceilings() -> Result<(), HalError> {
        for &spec in all_hardware_specs() {
            let computed = calculate_tpp(spec.mac_tops, spec.bit_length)?;
            assert_eq!(computed, spec.tpp_per_unit);

            let max_allowed_tpp = total_tpp_for_units(spec.model, spec.max_units_under_12m_tpp)?;
            assert!(max_allowed_tpp < AGGREGATE_TPP_STATUTORY_CEILING);

            let breach_tpp = total_tpp_for_units(spec.model, spec.max_units_under_12m_tpp + 1)?;
            assert!(breach_tpp >= AGGREGATE_TPP_STATUTORY_CEILING);
        }

        // Exact statutory hardware lookup table assertions
        assert_eq!(hardware_spec(AcceleratorModel::NvidiaH100Sxm).tpp_per_unit, 15_832);
        assert_eq!(hardware_spec(AcceleratorModel::NvidiaH100Sxm).max_units_under_12m_tpp, 757);
        assert_eq!(hardware_spec(AcceleratorModel::NvidiaB200Blackwell).tpp_per_unit, 36_000);
        assert_eq!(hardware_spec(AcceleratorModel::NvidiaB200Blackwell).max_units_under_12m_tpp, 333);
        assert_eq!(hardware_spec(AcceleratorModel::AmdInstinctMi300x).tpp_per_unit, 20_880);
        assert_eq!(hardware_spec(AcceleratorModel::AmdInstinctMi300x).max_units_under_12m_tpp, 574);
        assert_eq!(hardware_spec(AcceleratorModel::TenstorrentBlackholeP150).tpp_per_unit, 9_400);
        assert_eq!(hardware_spec(AcceleratorModel::FuriosaRngd).tpp_per_unit, 8_192);
        assert_eq!(hardware_spec(AcceleratorModel::FuriosaRngd).sriov_slice_count, 8);
        assert_eq!(hardware_spec(AcceleratorModel::FuriosaRngdSriovSlice).tpp_per_unit, 1_024);
        assert_eq!(
            hardware_spec(AcceleratorModel::FuriosaRngdSriovSlice).tpp_per_unit * 8,
            hardware_spec(AcceleratorModel::FuriosaRngd).tpp_per_unit
        );
        assert_eq!(
            hardware_spec(AcceleratorModel::FuriosaRngdSriovSlice).vram_total_bytes * 8,
            hardware_spec(AcceleratorModel::FuriosaRngd).vram_total_bytes
        );
        Ok(())
    }

    #[test]
    fn calculate_tpp_rejects_invalid_and_overflow_inputs() {
        assert!(matches!(
            calculate_tpp(0.0, 8),
            Err(HalError::InvalidTppParameters { .. })
        ));
        assert!(matches!(
            calculate_tpp(-100.0, 8),
            Err(HalError::InvalidTppParameters { .. })
        ));
        assert!(matches!(
            calculate_tpp(f64::NAN, 8),
            Err(HalError::InvalidTppParameters { .. })
        ));
        assert!(matches!(
            calculate_tpp(f64::INFINITY, 8),
            Err(HalError::InvalidTppParameters { .. })
        ));
        assert!(matches!(
            calculate_tpp(500.0, 0),
            Err(HalError::InvalidTppParameters { .. })
        ));
        assert!(matches!(
            calculate_tpp(1e30, 8),
            Err(HalError::ArithmeticOverflow { .. })
        ));
    }

    #[test]
    fn resolve_k8s_resources_for_all_supported_vendors() -> Result<(), HalError> {
        assert_eq!(
            resolve_model_from_k8s_resource("nvidia.com/h100", None)?,
            AcceleratorModel::NvidiaH100Sxm
        );
        assert_eq!(
            resolve_model_from_k8s_resource("nvidia.com/b200", None)?,
            AcceleratorModel::NvidiaB200Blackwell
        );
        assert_eq!(
            resolve_model_from_k8s_resource("nvidia.com/gpu", Some("B200"))?,
            AcceleratorModel::NvidiaB200Blackwell
        );
        assert_eq!(
            resolve_model_from_k8s_resource("nvidia.com/gpu", None)?,
            AcceleratorModel::NvidiaH100Sxm
        );
        assert_eq!(
            resolve_model_from_k8s_resource("amd.com/mi300x", None)?,
            AcceleratorModel::AmdInstinctMi300x
        );
        assert_eq!(
            resolve_model_from_k8s_resource("tenstorrent.com/blackhole-p150", None)?,
            AcceleratorModel::TenstorrentBlackholeP150
        );
        assert_eq!(
            resolve_model_from_k8s_resource("furiosa.ai/rngd", None)?,
            AcceleratorModel::FuriosaRngd
        );
        assert_eq!(
            resolve_model_from_k8s_resource("furiosa.ai/rngd-sriov", None)?,
            AcceleratorModel::FuriosaRngdSriovSlice
        );
        assert!(matches!(
            resolve_model_from_k8s_resource("vendor.invalid/tpu", None),
            Err(HalError::UnknownHardwareModel { .. })
        ));
        assert!(matches!(
            resolve_model_from_k8s_resource("nvidia.com/gpu", Some("unknown-sku")),
            Err(HalError::UnknownHardwareModel { .. })
        ));
        Ok(())
    }
}
