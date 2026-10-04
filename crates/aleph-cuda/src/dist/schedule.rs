//! Splits an exchange's chunk pairs into scratch-sized pieces and packs them
//! into rounds. One round = one NCCL group: every piece's two halves are
//! received into scratch slots, then copied back into the rank slices. The
//! scratch is the only extra memory, so a rank slice keeps its full reach.

use super::exchange::ChunkPair;

/// One piece of a chunk-pair swap: rank `ra`'s amplitudes `[a_off, a_off+len)`
/// trade places with rank `rb`'s `[b_off, b_off+len)`. `b`'s half lands in
/// scratch slot `slot_a` on device `da`, `a`'s half in `slot_b` on device `db`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Piece {
    pub ra: u32,
    pub a_off: usize,
    pub da: usize,
    pub slot_a: usize,
    pub rb: u32,
    pub b_off: usize,
    pub db: usize,
    pub slot_b: usize,
    pub len: usize,
}

/// Split every chunk pair into pieces of `min(chunk, cap/2)` amplitudes and
/// pack them greedily into rounds in which each device uses at most `cap`
/// scratch amplitudes. `cap` is a power of two >= 2; the `cap/2` piece lets a
/// same-device pair (both halves on one device) fit an empty round.
pub(crate) fn schedule(
    pairs: &[ChunkPair],
    chunk: usize,
    cap: usize,
    dev_of: impl Fn(u32) -> usize,
    n_dev: usize,
) -> Vec<Vec<Piece>> {
    let len = chunk.min((cap / 2).max(1));
    let mut rounds = Vec::new();
    let mut cur: Vec<Piece> = Vec::new();
    let mut used = vec![0usize; n_dev];
    for &((ra, ca), (rb, cb)) in pairs {
        let (da, db) = (dev_of(ra), dev_of(rb));
        let mut off = 0;
        while off < chunk {
            let need_a = if da == db { 2 * len } else { len };
            if used[da] + need_a > cap || used[db] + len > cap {
                rounds.push(std::mem::take(&mut cur));
                used.iter_mut().for_each(|u| *u = 0);
            }
            let slot_a = used[da];
            used[da] += len;
            let slot_b = used[db];
            used[db] += len;
            cur.push(Piece {
                ra,
                a_off: ca as usize * chunk + off,
                da,
                slot_a,
                rb,
                b_off: cb as usize * chunk + off,
                db,
                slot_b,
                len,
            });
            off += len;
        }
    }
    if !cur.is_empty() {
        rounds.push(cur);
    }
    rounds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::exchange::{chunk_pairs, rank_device};
    use aleph_ir::dist::DistLayout;

    /// Every amplitude of every moved chunk is in exactly one piece; per
    /// round, each device's scratch slots are disjoint and inside `cap`.
    #[test]
    fn schedule_covers_once_and_respects_scratch() {
        for (n, g, d) in [(8u32, 1u32, 1usize), (8, 2, 2), (10, 3, 4), (10, 3, 8)] {
            let l = DistLayout::new(n, g).unwrap();
            let m = l.m();
            for bits in [
                vec![m],
                vec![n - 1],
                vec![m, n - 1],
                (m..n).collect::<Vec<_>>(),
            ] {
                let k = bits.len() as u32;
                if k > m || (k == 2 && bits[0] == bits[1]) {
                    continue;
                }
                let chunk = 1usize << (m - k);
                for cap in [2usize, 8, 1 << 20] {
                    let pairs = chunk_pairs(l, &bits);
                    let rounds = schedule(&pairs, chunk, cap, |r| rank_device(r, l.ranks(), d), d);
                    let mut covered = std::collections::HashSet::new();
                    for round in &rounds {
                        assert!(!round.is_empty());
                        let mut used: Vec<Vec<(usize, usize)>> = vec![vec![]; d];
                        for p in round {
                            for i in 0..p.len {
                                assert!(covered.insert((p.ra, p.a_off + i)), "a dup");
                                assert!(covered.insert((p.rb, p.b_off + i)), "b dup");
                            }
                            assert_eq!(p.da, rank_device(p.ra, l.ranks(), d));
                            assert_eq!(p.db, rank_device(p.rb, l.ranks(), d));
                            used[p.da].push((p.slot_a, p.len));
                            used[p.db].push((p.slot_b, p.len));
                        }
                        for slots in &mut used {
                            slots.sort_unstable();
                            let mut end = 0;
                            for &(s, len) in slots.iter() {
                                assert!(s >= end && s + len <= cap, "slot overlap / over cap");
                                end = s + len;
                            }
                        }
                    }
                    let moved = (l.ranks() as usize) * ((1usize << k) - 1) * chunk;
                    assert_eq!(covered.len(), moved, "n={n} g={g} d={d} {bits:?} cap={cap}");
                }
            }
        }
    }
}
