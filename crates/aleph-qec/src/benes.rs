//! Beneš rearrangeable permutation network routing.
//!
//! A Beneš network on M = 2^k inputs realises any permutation with 2·log2(M)−1
//! columns of M/2 2×2 switches. `benes_control` returns the switch settings for a
//! given bijection; `benes_apply` simulates the fabric (test / gen-time guard oracle).
//! Ref: Beneš (1965); Lee, "On the Rearrangeability of 2(log2 N)−1 Stage Permutation
//! Networks", IEEE ToC C-34(5), 1985 (looping algorithm).
//!
//! Implementation note: `route` (control-bit synthesis) and `benes_apply` (fabric
//! simulation) share one recursive block decomposition — same column/row addressing,
//! same bar/cross convention — so they agree by construction rather than by
//! independently re-deriving a butterfly wiring formula that has to be kept in sync
//! by hand. `random_bijections` below is the round-trip oracle that proves it.

/// Number of switch columns in a Beneš network on `m` inputs: `2*log2(m) - 1`.
pub fn benes_columns(m: usize) -> usize {
    assert!(
        m.is_power_of_two() && m >= 2,
        "Beneš size must be a power of two >= 2"
    );
    2 * (m.trailing_zeros() as usize) - 1
}

/// Extend a partial injection (`dest[i] = Some(b)` routes input `i` to output `b`;
/// `None` marks a padding input) into a full bijection on `0..m`, filling unused
/// inputs with the unused outputs in ascending order.
///
/// Panics if `dest` assigns two inputs to the same output (violates the
/// ≤1-per-bank invariant the caller relies on).
pub fn complete_partial(dest: &[Option<usize>], m: usize) -> Vec<usize> {
    assert_eq!(dest.len(), m);
    let mut used = vec![false; m];
    let mut out = vec![usize::MAX; m];
    for (i, d) in dest.iter().enumerate() {
        if let Some(b) = *d {
            assert!(b < m, "target {b} out of range {m}");
            assert!(
                !used[b],
                "duplicate Beneš target {b} (violates <=1-per-bank invariant)"
            );
            used[b] = true;
            out[i] = b;
        }
    }
    let mut free = (0..m).filter(|&b| !used[b]);
    for slot in out.iter_mut() {
        if *slot == usize::MAX {
            *slot = free
                .next()
                .expect("free list underflow: dest not an injection");
        }
    }
    out
}

/// Switch control bits, column-major (`ctrl[col*(m/2) + switch]`): `false` = bar
/// (straight through), `true` = cross.
pub fn benes_control(perm: &[usize]) -> Vec<bool> {
    let m = perm.len();
    assert!(
        m.is_power_of_two() && m >= 2,
        "Beneš size must be a power of two >= 2"
    );
    let cols = benes_columns(m);
    let mut ctrl = vec![false; cols * (m / 2)];
    route(perm, &mut ctrl, 0, 0, m / 2);
    ctrl
}

/// Recursive control-bit synthesis (Lee's looping algorithm). `perm` is a bijection
/// on `0..n` (n = 2^t, the size of the block being routed). `col0` is this block's
/// input-column index in the global `ctrl`; `row0` is the row (switch index within a
/// column) at which this block's `n/2` switches start; `stride` = M/2 is the global,
/// constant number of switches per column.
fn route(perm: &[usize], ctrl: &mut [bool], col0: usize, row0: usize, stride: usize) {
    let n = perm.len();
    if n == 2 {
        // Single column, single switch: bar if perm == [0,1], cross if [1,0].
        ctrl[col0 * stride + row0] = perm[0] == 1;
        return;
    }
    let half = n / 2;
    let mut inv = vec![0usize; n];
    for (i, &o) in perm.iter().enumerate() {
        inv[o] = i;
    }

    // 2-edge-colour the (input-pair, output-pair) request graph: local edge e (input e
    // -> output perm[e]) gets colour true="upper subnet" / false="lower subnet" such
    // that the two edges touching any input-pair, and the two touching any
    // output-pair, always differ. That graph is 2-regular bipartite, i.e. a disjoint
    // union of even cycles; walk each cycle, alternating which node's edges are
    // pinned, until it closes (Beneš/Lee looping algorithm).
    let mut color: Vec<Option<bool>> = vec![None; n];
    for e0 in 0..n {
        if color[e0].is_some() {
            continue;
        }
        let mut e = e0;
        loop {
            color[e] = Some(true);
            // The other edge sharing e's output-pair must take the opposite colour.
            let q = perm[e] / 2;
            let out_pair = [inv[2 * q], inv[2 * q + 1]];
            let e2 = if out_pair[0] == e {
                out_pair[1]
            } else {
                out_pair[0]
            };
            if color[e2].is_some() {
                break;
            }
            color[e2] = Some(false);
            // The other edge sharing e2's input-pair must take the opposite colour
            // (i.e. true again — it becomes the next "true" pivot in this cycle).
            let p2 = e2 / 2;
            let in_pair = [2 * p2, 2 * p2 + 1];
            let e3 = if in_pair[0] == e2 {
                in_pair[1]
            } else {
                in_pair[0]
            };
            if color[e3].is_some() {
                break;
            }
            e = e3;
        }
    }
    let color: Vec<bool> = color
        .into_iter()
        .map(|c| c.expect("uncoloured edge"))
        .collect();

    // Build the two half-size sub-permutations and this block's input/output columns.
    let mut up = vec![0usize; half];
    let mut lo = vec![0usize; half];
    for isw in 0..half {
        let (e_up, e_lo) = if color[2 * isw] {
            (2 * isw, 2 * isw + 1)
        } else {
            (2 * isw + 1, 2 * isw)
        };
        up[isw] = perm[e_up] / 2;
        lo[isw] = perm[e_lo] / 2;
        // false(bar): even local input -> upper subnet; true(cross): odd -> upper.
        ctrl[col0 * stride + row0 + isw] = !color[2 * isw];
    }
    let out_col = col0 + benes_columns(n) - 1;
    for osw in 0..half {
        let out_upper = color[inv[2 * osw]];
        // false(bar): upper subnet -> even local output pin; true(cross): -> odd.
        ctrl[out_col * stride + row0 + osw] = !out_upper;
    }

    route(&up, ctrl, col0 + 1, row0, stride);
    route(&lo, ctrl, col0 + 1, row0 + half / 2, stride);
}

/// Simulate the fabric with `ctrl` on `input` (length M). Returns `output[o]` = the
/// value routed to output `o`. Test / gen-time guard oracle — not on the RTL path.
pub fn benes_apply(ctrl: &[bool], input: &[usize]) -> Vec<usize> {
    let m = input.len();
    assert!(
        m.is_power_of_two() && m >= 2,
        "Beneš size must be a power of two >= 2"
    );
    apply_block(ctrl, input, 0, 0, m / 2)
}

/// Mirrors `route`'s recursive block decomposition exactly (same column/row
/// addressing, same bar/cross convention) so it agrees with the synthesised control
/// bits by construction.
fn apply_block(
    ctrl: &[bool],
    input: &[usize],
    col0: usize,
    row0: usize,
    stride: usize,
) -> Vec<usize> {
    let n = input.len();
    if n == 2 {
        return if ctrl[col0 * stride + row0] {
            vec![input[1], input[0]]
        } else {
            vec![input[0], input[1]]
        };
    }
    let half = n / 2;
    let mut upper_in = vec![0usize; half];
    let mut lower_in = vec![0usize; half];
    for isw in 0..half {
        if ctrl[col0 * stride + row0 + isw] {
            upper_in[isw] = input[2 * isw + 1];
            lower_in[isw] = input[2 * isw];
        } else {
            upper_in[isw] = input[2 * isw];
            lower_in[isw] = input[2 * isw + 1];
        }
    }
    let out_col = col0 + benes_columns(n) - 1;
    let upper_out = apply_block(ctrl, &upper_in, col0 + 1, row0, stride);
    let lower_out = apply_block(ctrl, &lower_in, col0 + 1, row0 + half / 2, stride);

    let mut out = vec![0usize; n];
    for osw in 0..half {
        if ctrl[out_col * stride + row0 + osw] {
            out[2 * osw] = lower_out[osw];
            out[2 * osw + 1] = upper_out[osw];
        } else {
            out[2 * osw] = upper_out[osw];
            out[2 * osw + 1] = lower_out[osw];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Apply control bits, assert the network realises exactly `perm`.
    fn check(perm: &[usize]) {
        let ctrl = benes_control(perm);
        assert_eq!(ctrl.len(), benes_columns(perm.len()) * (perm.len() / 2));
        let ident: Vec<usize> = (0..perm.len()).collect();
        let routed = benes_apply(&ctrl, &ident);
        // input i must land at output perm[i]  =>  routed[perm[i]] == i
        for (i, &o) in perm.iter().enumerate() {
            assert_eq!(
                routed[o], i,
                "perm {:?}: output {} got input {} not {}",
                perm, o, routed[o], i
            );
        }
    }

    #[test]
    fn identity_8() {
        check(&[0, 1, 2, 3, 4, 5, 6, 7]);
    }
    #[test]
    fn reverse_8() {
        check(&[7, 6, 5, 4, 3, 2, 1, 0]);
    }
    #[test]
    fn swap_pairs_8() {
        check(&[1, 0, 3, 2, 5, 4, 7, 6]);
    }

    #[test]
    fn random_bijections() {
        // deterministic LCG (no dev-dep); cover the production sizes.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for &m in &[2usize, 4, 8, 16, 512, 1024] {
            for _ in 0..64 {
                let mut p: Vec<usize> = (0..m).collect();
                for i in (1..m).rev() {
                    let j = (rng() as usize) % (i + 1);
                    p.swap(i, j);
                }
                check(&p);
            }
        }
    }

    #[test]
    fn partial_completes_and_routes() {
        // 3 real sources into 8 outputs (banks): s0->5, s1->2, s2->6
        let dest = vec![Some(5), Some(2), Some(6), None, None, None, None, None];
        let full = complete_partial(&dest, 8);
        assert_eq!(full[0], 5);
        assert_eq!(full[1], 2);
        assert_eq!(full[2], 6);
        // Unused outputs {0,1,3,4,7} must fill inputs 3..8 in ascending order.
        assert_eq!(full, vec![5, 2, 6, 0, 1, 3, 4, 7]);
        assert_eq!(
            full.iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            8
        );
        check(&full);
    }

    #[test]
    #[should_panic]
    fn duplicate_target_rejected() {
        complete_partial(&[Some(3), Some(3), None, None], 4);
    }

    #[test]
    fn complete_partial_non_power_of_two() {
        // m = 3 is not a power of two: unused output {1} fills the None input.
        assert_eq!(
            complete_partial(&[Some(2), None, Some(0)], 3),
            vec![2, 1, 0]
        );
    }
}
