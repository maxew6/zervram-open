use std::collections::BTreeSet;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use zervram_hal::{AcceleratorDevice, MemoryRange};

use crate::error::ScrubError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum SamplingStrategy {
    /// Verifies 100% of mapped/captured 4 KiB page frames returned by the HAL.
    DeterministicAllMapped,
    /// Checks first page, last page, and `random_interior_pages` CSPRNG-sampled 4 KiB pages.
    StratifiedSpotSample {
        random_interior_pages: usize,
        include_boundary_pages: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageVerificationReport {
    pub sampled_pages: usize,
    pub dirty_pages_found: usize,
    pub first_dirty_page_addr: Option<u64>,
    pub first_dirty_non_zero_bytes: usize,
    pub verification_duration_us: u64,
    pub all_zero: bool,
}

fn build_stratified_page_addrs(
    range: MemoryRange,
    random_interior_pages: usize,
    include_boundary_pages: bool,
) -> Vec<u64> {
    let total_pages = range.page_count();
    if total_pages == 0 {
        return Vec::new();
    }

    let page_size = u64::from(range.page_size_bytes);
    let mut selected = BTreeSet::new();

    if include_boundary_pages {
        selected.insert(range.start_addr);
        if let Some(last_offset) = total_pages.saturating_sub(1).checked_mul(page_size) {
            if let Some(last_addr) = range.start_addr.checked_add(last_offset) {
                selected.insert(last_addr);
            }
        }
    }

    if total_pages > 2 && random_interior_pages > 0 {
        let mut rng = ChaCha20Rng::from_entropy();
        let target_interior = random_interior_pages.min(
            usize::try_from(total_pages.saturating_sub(2)).unwrap_or(random_interior_pages),
        );
        let max_attempts = target_interior.saturating_mul(4).max(16);
        for _ in 0..max_attempts {
            if selected.len()
                >= target_interior.saturating_add(usize::from(include_boundary_pages) * 2)
            {
                break;
            }
            let page_idx = rng.gen_range(1..total_pages.saturating_sub(1));
            if let Some(offset) = page_idx.checked_mul(page_size) {
                if let Some(addr) = range.start_addr.checked_add(offset) {
                    selected.insert(addr);
                }
            }
        }
    } else if !include_boundary_pages && random_interior_pages > 0 {
        selected.insert(range.start_addr);
    }

    selected.into_iter().collect()
}

/// Reads back 4 KiB device pages according to `strategy` and verifies every sampled byte equals `0x00`.
/// Fails closed with `ScrubError::EmptyVerificationSample` if 0 pages would be checked.
pub async fn verify_zeroized_pages(
    hal: &dyn AcceleratorDevice,
    device_uuid: Uuid,
    range: MemoryRange,
    strategy: SamplingStrategy,
) -> Result<PageVerificationReport, ScrubError> {
    let started = Instant::now();

    let page_addrs = match strategy {
        SamplingStrategy::DeterministicAllMapped => {
            match hal.mapped_verification_pages(device_uuid, range).await? {
                Some(mapped) if !mapped.is_empty() => mapped,
                _ => build_stratified_page_addrs(range, 64, true),
            }
        }
        SamplingStrategy::StratifiedSpotSample {
            random_interior_pages,
            include_boundary_pages,
        } => build_stratified_page_addrs(range, random_interior_pages, include_boundary_pages),
    };

    if page_addrs.is_empty() {
        return Err(ScrubError::EmptyVerificationSample { device_uuid });
    }

    let samples = hal.read_back_pages(device_uuid, &page_addrs).await?;
    if samples.is_empty() || samples.len() != page_addrs.len() {
        return Err(ScrubError::EmptyVerificationSample { device_uuid });
    }

    let mut dirty_pages_found: usize = 0;
    let mut first_dirty_page_addr: Option<u64> = None;
    let mut first_dirty_non_zero_bytes: usize = 0;

    for sample in &samples {
        if !sample.is_all_zero() {
            dirty_pages_found = dirty_pages_found.saturating_add(1);
            if first_dirty_page_addr.is_none() {
                first_dirty_page_addr = Some(sample.page_addr);
                first_dirty_non_zero_bytes = sample.non_zero_byte_count();
            }
        }
    }

    let verification_duration_us =
        u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);

    Ok(PageVerificationReport {
        sampled_pages: samples.len(),
        dirty_pages_found,
        first_dirty_page_addr,
        first_dirty_non_zero_bytes,
        verification_duration_us,
        all_zero: !samples.is_empty() && dirty_pages_found == 0,
    })
}