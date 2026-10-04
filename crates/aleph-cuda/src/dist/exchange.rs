//! Rank-slice exchange transports.
//!
//! A `DistStep::Exchange { global_bits }` swaps global bit `global_bits[j]`
//! with local bit `m-k+j`. On chunks (the top `k` local bits of a rank) that
//! is an involution `(r, c) ↔ (r', c')`, so every moved chunk pair is swapped
//! exactly once, in place, through a bounded scratch buffer — no second copy
//! of the state, so the reach of a rank slice is not halved.

use std::marker::PhantomData;

use aleph_backend::BackendError;
use aleph_ir::dist::DistLayout;

use super::DeviceSv;

/// Moves amplitudes between rank slices for a `DistStep::Exchange`.
///
/// `devs[d]` is the backend of device `d`; rank `r` lives on device
/// [`rank_device`]`(r, R, devs.len())`.
pub trait Exchange<B: DeviceSv> {
    fn exchange(
        &mut self,
        devs: &mut [B],
        ranks: &mut [B::State],
        layout: DistLayout,
        global_bits: &[u32],
    ) -> Result<(), BackendError>;
}

/// Device index of rank `r` when `ranks` ranks sit on `devs` devices in
/// contiguous blocks (`devs` | `ranks`, both powers of two, checked by the
/// caller): the top `log2(devs)` rank bits name the device.
pub(crate) fn rank_device(r: u32, ranks: u32, devs: usize) -> usize {
    let per = (ranks as usize / devs.max(1)).max(1);
    r as usize / per
}

/// Every unordered moved chunk pair `((r, c), (r', c'))` once, `(r,c) < (r',c')`.
pub(crate) fn chunk_pairs(l: DistLayout, global_bits: &[u32]) -> Vec<((u32, u32), (u32, u32))> {
    let m = l.m();
    let k = global_bits.len() as u32;
    let mut out = Vec::new();
    for r in 0..l.ranks() {
        for c in 0..(1u32 << k) {
            let mut r2 = r;
            let mut c2 = 0u32;
            for (j, &gb) in global_bits.iter().enumerate() {
                let rb = gb - m;
                c2 |= ((r >> rb) & 1) << j;
                r2 = (r2 & !(1 << rb)) | (((c >> j) & 1) << rb);
            }
            if (r, c) < (r2, c2) {
                out.push(((r, c), (r2, c2)));
            }
        }
    }
    out
}

/// Disjoint `&mut` to two different ranks (`None` if equal or out of range).
pub(crate) fn two_mut<T>(v: &mut [T], a: usize, b: usize) -> Option<(&mut T, &mut T)> {
    if a == b || a.max(b) >= v.len() {
        return None;
    }
    if a < b {
        let (lo, hi) = v.split_at_mut(b);
        Some((&mut lo[a], &mut hi[0]))
    } else {
        let (lo, hi) = v.split_at_mut(a);
        Some((&mut hi[0], &mut lo[b]))
    }
}

/// Every bit global, in range, and no bit named twice (a repeat would make
/// `chunk_pairs` pair a rank with itself or visit a chunk twice).
pub(crate) fn valid_bits(l: DistLayout, global_bits: &[u32]) -> bool {
    let mut seen = 0u64;
    for &b in global_bits {
        if b >= u64::BITS || !l.is_global(b) || b >= l.n || seen & (1u64 << b) != 0 {
            return false;
        }
        seen |= 1u64 << b;
    }
    true
}

/// Ranks on devices that share one CUDA ordinal (one GPU, possibly several
/// backends): chunk swaps are device-to-device copies through a per-device
/// scratch of at most `scratch_amps` amplitudes. A pair whose two devices are
/// different GPUs is rejected: that needs `NcclExchange`.
pub struct LocalExchange<B: DeviceSv> {
    scratch_amps: usize,
    /// Per device index: (scratch slice, amplitudes it holds).
    scratch: Vec<Option<(B::State, usize)>>,
    _b: PhantomData<B>,
}

impl<B: DeviceSv> LocalExchange<B> {
    /// 2^24 amplitudes = 256 MiB of FP64 complex scratch.
    pub const DEFAULT_SCRATCH_AMPS: usize = 1 << 24;

    pub fn new() -> Self {
        Self::with_scratch_amps(Self::DEFAULT_SCRATCH_AMPS)
    }

    /// Scratch of `amps` amplitudes (rounded up to a power of two, clamped to
    /// `1..=2^40` so the rounding cannot overflow).
    pub fn with_scratch_amps(amps: usize) -> Self {
        Self {
            scratch_amps: amps.clamp(1, 1 << 40).next_power_of_two(),
            scratch: Vec::new(),
            _b: PhantomData,
        }
    }
}

impl<B: DeviceSv> Default for LocalExchange<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: DeviceSv> Exchange<B> for LocalExchange<B> {
    fn exchange(
        &mut self,
        devs: &mut [B],
        ranks: &mut [B::State],
        l: DistLayout,
        global_bits: &[u32],
    ) -> Result<(), BackendError> {
        let m = l.m();
        let k = global_bits.len() as u32;
        if k == 0
            || k > m
            || !valid_bits(l, global_bits)
            || ranks.len() != l.ranks() as usize
            || devs.is_empty()
        {
            return Err(BackendError::InvalidState {
                reason: "dist: bad exchange bits",
            });
        }
        let chunk = 1usize << (m - k);
        let piece = chunk.min(self.scratch_amps);
        self.scratch.resize_with(devs.len(), || None);
        for ((ra, ca), (rb, cb)) in chunk_pairs(l, global_bits) {
            let da = rank_device(ra, l.ranks(), devs.len());
            let db = rank_device(rb, l.ranks(), devs.len());
            if devs[da].ordinal() != devs[db].ordinal() {
                return Err(BackendError::InvalidState {
                    reason: "dist: cross-GPU chunk pair needs NcclExchange",
                });
            }
            // Scratch lives on rank a's device; grow it when a later, smaller-k
            // exchange has bigger chunks (never beyond `scratch_amps`).
            let slot = &mut self.scratch[da];
            if slot.as_ref().is_none_or(|(_, len)| *len < piece) {
                *slot = Some((devs[da].alloc_rank(piece.trailing_zeros(), 1)?, piece));
            }
            let Some((scr, _)) = slot.as_mut() else {
                return Err(BackendError::InvalidState {
                    reason: "dist: scratch missing",
                });
            };
            let (a0, b0) = (ca as usize * chunk, cb as usize * chunk);
            let Some((sa, sb)) = two_mut(ranks, ra as usize, rb as usize) else {
                return Err(BackendError::InvalidState {
                    reason: "dist: exchange paired a rank with itself",
                });
            };
            // Same ordinal => same primary context and legacy default stream,
            // so issuing every copy through devs[da] keeps them stream-ordered.
            let be = &mut devs[da];
            let mut off = 0;
            while off < chunk {
                be.copy_amps(sa, a0 + off, scr, 0, piece)?;
                be.copy_amps(sb, b0 + off, sa, a0 + off, piece)?;
                be.copy_amps(scr, 0, sb, b0 + off, piece)?;
                off += piece;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_pairs_cover_each_moved_chunk_once() {
        let l = DistLayout::new(8, 2).unwrap(); // m = 6
        for bits in [vec![6u32], vec![7], vec![6, 7], vec![7, 6]] {
            let k = bits.len() as u32;
            let pairs = chunk_pairs(l, &bits);
            let mut seen = std::collections::HashSet::new();
            for (a, b) in &pairs {
                assert!(a < b);
                assert!(seen.insert(*a) && seen.insert(*b), "dup in {bits:?}");
            }
            // moved chunks = all (r,c) minus fixed points; fixed points per rank = 1
            let total = (l.ranks() as usize) << k;
            assert_eq!(seen.len(), total - l.ranks() as usize, "{bits:?}");
        }
    }
}
