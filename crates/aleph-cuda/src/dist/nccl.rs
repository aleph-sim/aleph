//! Cross-device chunk exchange through NCCL point-to-point (P6-01b, spec §3.3).
//!
//! One NCCL rank per device, all driven from this process (`ncclCommInitAll`).
//! An exchange is split by [`schedule`] into rounds that fit a per-device
//! scratch; a round is one `ncclGroupStart/End` in which each piece's halves
//! are sent from the rank slices and received into the peer's scratch,
//! followed by device-local scratch→slice copies. Comms are bound to each
//! backend's own stream, so NCCL is ordered after the preceding kernels and
//! before the following ones without a host barrier.

use std::marker::PhantomData;

use aleph_backend::BackendError;
use aleph_ir::dist::DistLayout;
use cudarc::driver::CudaViewMut;
use cudarc::nccl::result::NcclError;
use cudarc::nccl::safe::{group_end, group_start, Comm, NcclType};

use super::exchange::{chunk_pairs, rank_device, two_mut, valid_bits, valid_devices, Exchange};
use super::schedule::{schedule, Piece};
use super::DeviceSv;

fn nccl_err(_e: NcclError) -> BackendError {
    BackendError::InvalidState {
        reason: "nccl: communication failure",
    }
}

fn scratch_missing() -> BackendError {
    BackendError::InvalidState {
        reason: "dist: scratch missing",
    }
}

/// Chunk exchange across GPUs. Same-device pairs take the D2D-copy path of
/// `LocalExchange` unless [`NcclExchange::route_all_through_nccl`] is set.
pub struct NcclExchange<B: DeviceSv> {
    /// `comms[d]` is NCCL rank `d`, bound to `devs[d]`'s stream.
    comms: Vec<Comm>,
    scratch_amps: usize,
    /// Per device: one `scratch_amps`-amplitude slice, allocated on first use.
    scratch: Vec<Option<B::State>>,
    route_all: bool,
    _b: PhantomData<B>,
}

impl<B: DeviceSv> NcclExchange<B>
where
    B::Scalar: NcclType,
{
    /// 2^24 amplitudes = 256 MiB of FP64 complex scratch per device.
    pub const DEFAULT_SCRATCH_AMPS: usize = 1 << 24;

    /// One NCCL rank per backend, in `devs` order (pass the same order to
    /// `DistSvBackend::multi`). Ordinals must be distinct: NCCL rejects two
    /// ranks on one GPU. Errors (never panics) when libnccl is absent.
    pub fn new(devs: &[B]) -> Result<Self, BackendError> {
        let mut ords: Vec<usize> = devs.iter().map(DeviceSv::ordinal).collect();
        ords.sort_unstable();
        if ords.is_empty() || ords.windows(2).any(|w| w[0] == w[1]) {
            return Err(BackendError::InvalidState {
                reason: "nccl: need >=1 device and one backend per distinct GPU",
            });
        }
        // cudarc's dynamic loader panics on first use when the library is
        // missing; probe first so a host without NCCL gets an error.
        // SAFETY: `is_culib_present` only dlopens/dlcloses candidate names.
        if !unsafe { cudarc::nccl::sys::is_culib_present() } {
            return Err(BackendError::InvalidState {
                reason: "nccl: libnccl.so not loadable (cudarc dlopens the unversioned name: install libnccl-dev)",
            });
        }
        let comms =
            Comm::from_devices(devs.iter().map(DeviceSv::stream).collect()).map_err(nccl_err)?;
        Ok(Self {
            comms,
            scratch_amps: Self::DEFAULT_SCRATCH_AMPS,
            scratch: Vec::new(),
            route_all: false,
            _b: PhantomData,
        })
    }

    /// Per-device scratch of `amps` amplitudes (power of two, clamped to
    /// `2..=2^40`).
    pub fn with_scratch_amps(mut self, amps: usize) -> Self {
        self.scratch_amps = amps.clamp(2, 1 << 40).next_power_of_two();
        self.scratch.clear();
        self
    }

    /// Diagnostic: route same-device pairs through NCCL self send/recv too,
    /// so one GPU exercises the full NCCL piece protocol.
    pub fn route_all_through_nccl(mut self, on: bool) -> Self {
        self.route_all = on;
        self
    }
}

fn scratch_view_mut<B: DeviceSv>(
    scratch: &mut [Option<B::State>],
    d: usize,
    slot: usize,
    len: usize,
) -> Result<CudaViewMut<'_, B::Scalar>, BackendError> {
    let st = scratch
        .get_mut(d)
        .and_then(Option::as_mut)
        .ok_or_else(scratch_missing)?;
    B::amps_view_mut(st, slot, len)
}

/// Received halves: scratch → rank slices, on each receiving device's stream.
fn copy_back<B: DeviceSv>(
    devs: &mut [B],
    scratch: &[Option<B::State>],
    ranks: &mut [B::State],
    p: &Piece,
) -> Result<(), BackendError> {
    for (d, slot, r, off) in [
        (p.da, p.slot_a, p.ra, p.a_off),
        (p.db, p.slot_b, p.rb, p.b_off),
    ] {
        let scr = scratch
            .get(d)
            .and_then(Option::as_ref)
            .ok_or_else(scratch_missing)?;
        devs[d].copy_amps(scr, slot, &mut ranks[r as usize], off, p.len)?;
    }
    Ok(())
}

impl<B: DeviceSv> Exchange<B> for NcclExchange<B>
where
    B::Scalar: NcclType,
{
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
            || devs.len() != self.comms.len()
            || !valid_devices(devs.len(), l.ranks())
        {
            return Err(BackendError::InvalidState {
                reason: "dist: bad exchange bits",
            });
        }
        let chunk = 1usize << (m - k);
        let (nr, nd) = (l.ranks(), devs.len());
        let dev_of = |r: u32| rank_device(r, nr, nd);
        self.scratch.resize_with(nd, || None);
        for (d, slot) in self.scratch.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(devs[d].alloc_rank(self.scratch_amps.trailing_zeros(), 1)?);
            }
        }
        let route_all = self.route_all;
        let (nccl_pairs, local_pairs): (Vec<_>, Vec<_>) = chunk_pairs(l, global_bits)
            .into_iter()
            .partition(|&((ra, _), (rb, _))| route_all || dev_of(ra) != dev_of(rb));

        // Same-device pairs: three D2D copies through the scratch head.
        let piece = chunk.min(self.scratch_amps);
        for ((ra, ca), (rb, cb)) in local_pairs {
            let d = dev_of(ra);
            let scr = self.scratch[d].as_mut().ok_or_else(scratch_missing)?;
            let (a0, b0) = (ca as usize * chunk, cb as usize * chunk);
            let Some((sa, sb)) = two_mut(ranks, ra as usize, rb as usize) else {
                return Err(BackendError::InvalidState {
                    reason: "dist: exchange paired a rank with itself",
                });
            };
            let mut off = 0;
            while off < chunk {
                devs[d].copy_amps(sa, a0 + off, scr, 0, piece)?;
                devs[d].copy_amps(sb, b0 + off, sa, a0 + off, piece)?;
                devs[d].copy_amps(scr, 0, sb, b0 + off, piece)?;
                off += piece;
            }
        }

        // Cross-device (or all, with route_all) pairs: NCCL rounds. Send
        // sources are rank slices and recv targets are scratch slots the
        // scheduler hands out disjointly, so nothing in a group is both read
        // and written.
        for round in schedule(&nccl_pairs, chunk, self.scratch_amps, dev_of, nd) {
            group_start().map_err(nccl_err)?;
            let issued = (|| {
                for p in &round {
                    // comm rank == device index
                    let (ca, cb) = (&self.comms[p.da], &self.comms[p.db]);
                    let (peer_a, peer_b) = (p.db as i32, p.da as i32);
                    let send_a = B::amps_view(&ranks[p.ra as usize], p.a_off, p.len)?;
                    ca.send(&send_a, peer_a).map_err(nccl_err)?;
                    let send_b = B::amps_view(&ranks[p.rb as usize], p.b_off, p.len)?;
                    cb.send(&send_b, peer_b).map_err(nccl_err)?;
                    // NCCL matches sends and recvs between one (src, dst)
                    // pair in issue order. On a same-device piece both
                    // messages are d→d, so the recvs must follow the sends:
                    // a's half (sent first) → b's slot, then b's half → a's.
                    let mut rx = scratch_view_mut::<B>(&mut self.scratch, p.db, p.slot_b, p.len)?;
                    cb.recv(&mut rx, peer_b).map_err(nccl_err)?;
                    let mut rx = scratch_view_mut::<B>(&mut self.scratch, p.da, p.slot_a, p.len)?;
                    ca.recv(&mut rx, peer_a).map_err(nccl_err)?;
                }
                Ok::<(), BackendError>(())
            })();
            // Always close the group, even after a failed enqueue.
            let ended = group_end().map_err(nccl_err);
            issued?;
            ended?;
            for p in &round {
                copy_back(devs, &self.scratch, ranks, p)?;
            }
        }
        Ok(())
    }
}
