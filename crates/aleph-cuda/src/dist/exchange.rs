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
pub trait Exchange<B: DeviceSv> {
    fn exchange(
        &mut self,
        be: &mut B,
        ranks: &mut [B::State],
        layout: DistLayout,
        global_bits: &[u32],
    ) -> Result<(), BackendError>;
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

/// Disjoint `&mut` to two different ranks.
fn two_mut<T>(v: &mut [T], a: usize, b: usize) -> (&mut T, &mut T) {
    debug_assert_ne!(a, b);
    if a < b {
        let (lo, hi) = v.split_at_mut(b);
        (&mut lo[a], &mut hi[0])
    } else {
        let (lo, hi) = v.split_at_mut(a);
        (&mut hi[0], &mut lo[b])
    }
}

/// All ranks on one device: chunk swaps are device-to-device copies through a
/// scratch slice of at most `scratch_amps` amplitudes.
pub struct LocalExchange<B: DeviceSv> {
    scratch_amps: usize,
    scratch: Option<B::State>,
    /// Amplitudes the allocated scratch holds (0 = none yet).
    scratch_len: usize,
    _b: PhantomData<B>,
}

impl<B: DeviceSv> LocalExchange<B> {
    /// 2^24 amplitudes = 256 MiB of FP64 complex scratch.
    pub const DEFAULT_SCRATCH_AMPS: usize = 1 << 24;

    pub fn new() -> Self {
        Self::with_scratch_amps(Self::DEFAULT_SCRATCH_AMPS)
    }

    /// Scratch of `amps` amplitudes (rounded up to a power of two, min 1).
    pub fn with_scratch_amps(amps: usize) -> Self {
        Self {
            scratch_amps: amps.max(1).next_power_of_two(),
            scratch: None,
            scratch_len: 0,
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
        be: &mut B,
        ranks: &mut [B::State],
        l: DistLayout,
        global_bits: &[u32],
    ) -> Result<(), BackendError> {
        let m = l.m();
        let k = global_bits.len() as u32;
        if k == 0 || k > m || global_bits.iter().any(|&b| !l.is_global(b) || b >= l.n) {
            return Err(BackendError::InvalidState {
                reason: "dist: bad exchange bits",
            });
        }
        let chunk = 1usize << (m - k);
        let piece = chunk.min(self.scratch_amps);
        // A later exchange with smaller k has bigger chunks: grow the scratch
        // when the piece no longer fits (never beyond `scratch_amps`).
        if self.scratch_len < piece {
            self.scratch = Some(be.alloc_rank(piece.trailing_zeros(), 1)?);
            self.scratch_len = piece;
        }
        let Some(scr) = self.scratch.as_mut() else {
            return Err(BackendError::InvalidState {
                reason: "dist: scratch missing",
            });
        };
        for ((ra, ca), (rb, cb)) in chunk_pairs(l, global_bits) {
            let (a0, b0) = (ca as usize * chunk, cb as usize * chunk);
            let (sa, sb) = two_mut(ranks, ra as usize, rb as usize);
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
