use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use rand::seq::index::sample as sample_indices;
use rand::{Rng, RngCore};
use rand_chacha::ChaCha8Rng;
use uuid::Uuid;
use zeroize::Zeroize;

use crate::error::HalError;
use crate::model::{MemoryRange, PageSample, VRAM_PAGE_SIZE_BYTES};

#[derive(Debug)]
pub struct SimulatedVramPool {
    pub active_range: MemoryRange,
    /// Real host-RAM 4 KiB page frames keyed by virtual VRAM page address.
    /// Cleared with `zeroize::Zeroize` to prevent compiler dead-store elimination without `unsafe`.
    pub pages: BTreeMap<u64, Box<[u8; VRAM_PAGE_SIZE_BYTES as usize]>>,
}

impl SimulatedVramPool {
    /// Allocates `backing_pages` real 4 KiB host-RAM page frames evenly distributed across
    /// `active_range`, always covering the first and last boundary pages when `backing_pages >= 2`.
    pub fn new(active_range: MemoryRange, backing_pages: usize) -> Self {
        let total_pages = active_range.page_count().max(1);
        let requested = u64::try_from(backing_pages.max(1)).unwrap_or(total_pages);
        let target_count = requested.min(total_pages);

        let mut pages = BTreeMap::new();
        if target_count == 1 {
            pages.insert(
                active_range.start_addr,
                Box::new([0u8; VRAM_PAGE_SIZE_BYTES as usize]),
            );
        } else {
            let page_size = u64::from(active_range.page_size_bytes);
            let max_index = u128::from(total_pages.saturating_sub(1));
            let denom = u128::from(target_count.saturating_sub(1));

            for step in 0..target_count {
                let page_idx_u128 = (u128::from(step) * max_index) / denom;
                let page_idx = u64::try_from(page_idx_u128).unwrap_or(0);
                if let Some(offset) = page_idx.checked_mul(page_size) {
                    if let Some(addr) = active_range.start_addr.checked_add(offset) {
                        pages.insert(addr, Box::new([0u8; VRAM_PAGE_SIZE_BYTES as usize]));
                    }
                }
            }
        }

        Self {
            active_range,
            pages,
        }
    }

    /// Populates all mapped 4 KiB page frames with deterministic non-zero tenant weight/prompt data.
    pub fn taint_for_tenant(
        &mut self,
        rng: &mut ChaCha8Rng,
        tenant_id_hash: &str,
        job_id: &str,
    ) {
        let mut fingerprint_seed: u8 = 0x5A;
        for &b in tenant_id_hash.as_bytes().iter().chain(job_id.as_bytes()) {
            fingerprint_seed = fingerprint_seed.wrapping_add(b).rotate_left(1);
        }
        if fingerprint_seed == 0 {
            fingerprint_seed = 0xA5;
        }

        for page in self.pages.values_mut() {
            rng.fill_bytes(page.as_mut_slice());
            if let Some(first) = page.first_mut() {
                *first |= 0x80;
            }
            for (dst, &src) in page
                .iter_mut()
                .skip(1)
                .zip(tenant_id_hash.as_bytes().iter().take(32))
            {
                *dst = src ^ fingerprint_seed;
            }
        }
    }

    /// Synthetic test retention model for `mock-gpu`: clears most pages on release while retaining
    /// `min_pct..=max_pct` dirty pages (default `12..=18%` is an arbitrary synthetic test
    /// parameter for CI/unit testing, NOT an empirical measurement on NVIDIA HBM/GDDR hardware).
    pub fn simulate_leftover_locals_on_release(
        &mut self,
        rng: &mut ChaCha8Rng,
        min_pct: u8,
        max_pct: u8,
    ) -> u8 {
        let low = min_pct.min(100);
        let high = max_pct.clamp(low, 100);
        let dirty_pct = rng.gen_range(low..=high);

        let total = self.pages.len();
        if total == 0 {
            return dirty_pct;
        }

        let dirty_target = if dirty_pct == 0 {
            0
        } else {
            total
                .saturating_mul(usize::from(dirty_pct))
                .div_ceil(100)
                .clamp(1, total)
        };

        let retained_indices: BTreeSet<usize> = sample_indices(rng, total, dirty_target)
            .into_iter()
            .collect();

        for (idx, page) in self.pages.values_mut().enumerate() {
            if retained_indices.contains(&idx) {
                if page.iter().all(|&b| b == 0) {
                    if let Some(first) = page.first_mut() {
                        *first = 0xC3;
                    }
                }
            } else {
                page.as_mut_slice().zeroize();
            }
        }

        dirty_pct
    }

    /// Zeroizes every mapped 4 KiB page inside `range` using `zeroize::Zeroize`,
    /// returning `(elapsed_us, pages_zeroed)`.
    pub fn zeroize_range(&mut self, range: MemoryRange) -> Result<(u64, u64), HalError> {
        if !self.active_range.contains_range(&range) {
            return Err(HalError::InvalidMemoryRange {
                start_addr: range.start_addr,
                byte_len: range.byte_len,
                page_size: range.page_size_bytes,
            });
        }

        let started = Instant::now();
        let mut pages_zeroed: u64 = 0;

        for (&page_addr, page) in &mut self.pages {
            if range.contains_page(page_addr) {
                page.as_mut_slice().zeroize();
                pages_zeroed = pages_zeroed.saturating_add(1);
            }
        }

        let duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
        Ok((duration_us, pages_zeroed))
    }

    pub fn read_pages(
        &self,
        device_uuid: Uuid,
        page_addrs: &[u64],
    ) -> Result<Vec<PageSample>, HalError> {
        let mut samples = Vec::with_capacity(page_addrs.len());
        for &page_addr in page_addrs {
            if !self.active_range.contains_page(page_addr) {
                return Err(HalError::PageAddressOutOfBounds {
                    device_uuid,
                    page_addr,
                });
            }
            let data = match self.pages.get(&page_addr) {
                Some(mapped) => mapped.clone(),
                None => Box::new([0u8; VRAM_PAGE_SIZE_BYTES as usize]),
            };
            samples.push(PageSample { page_addr, data });
        }
        Ok(samples)
    }

    pub fn mapped_pages_in_range(&self, range: MemoryRange) -> Result<Vec<u64>, HalError> {
        if !self.active_range.contains_range(&range) {
            return Err(HalError::InvalidMemoryRange {
                start_addr: range.start_addr,
                byte_len: range.byte_len,
                page_size: range.page_size_bytes,
            });
        }
        Ok(self
            .pages
            .keys()
            .copied()
            .filter(|&addr| range.contains_page(addr))
            .collect())
    }

    pub fn dirty_page_count(&self) -> usize {
        self.pages
            .values()
            .filter(|page| page.iter().any(|&b| b != 0))
            .count()
    }

    pub fn inject_byte(
        &mut self,
        device_uuid: Uuid,
        page_addr: u64,
        byte_offset: usize,
        value: u8,
    ) -> Result<(), HalError> {
        let page = self
            .pages
            .get_mut(&page_addr)
            .ok_or(HalError::PageAddressOutOfBounds {
                device_uuid,
                page_addr,
            })?;
        let slot = page
            .get_mut(byte_offset)
            .ok_or(HalError::PageAddressOutOfBounds {
                device_uuid,
                page_addr,
            })?;
        *slot = value;
        Ok(())
    }
}