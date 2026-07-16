//! Arbitrary-size (AS-)Waksman rearrangeable permutation-network routing.
//!
//! A switch-optimal generalisation of the Beneš network to *any* number of inputs
//! `N` (not just powers of two). On `N` inputs it realises any permutation with
//! `ceil(N*log2 N) - N + 1` 2×2 switches — one fewer per recursion level than the
//! Beneš/Waksman power-of-two case, via Waksman's fixed-bypass optimisation.
//! Ref: Beauquier & Darrot, "On Arbitrary Size Waksman Networks", 2002; Waksman,
//! "A Permutation Network", J. ACM 15(1), 1968.
//!
//! # Recursion & the flat control layout (hard contract for the RTL)
//!
//! For a block of `n` inputs (`n >= 2`):
//!   * an **input stage** of `floor(n/2)` switches on pairs `(2i, 2i+1)`; if `n` is
//!     odd the last input `n-1` bypasses straight into the *upper* subnet;
//!   * an **upper subnet** of size `ceil(n/2)` and a **lower subnet** of size
//!     `floor(n/2)`;
//!   * an **output stage** of `ceil(n/2) - 1` switches on output pairs `(2j, 2j+1)`.
//!     The last output(s) are *hardwired* (this is the removed Waksman switch):
//!     for even `n` the upper subnet's last output feeds output `n-2` and the lower
//!     subnet's last feeds `n-1`; for odd `n` the upper subnet's last output feeds
//!     `n-1`.
//!
//! Control bits are a flat `Vec<bool>` (`false` = bar / straight, `true` = cross)
//! indexed by a **running switch counter in recursion order**:
//!
//! ```text
//!   [ input switches of this block ]
//!   [ ... entire upper subnet, recursively ... ]
//!   [ ... entire lower subnet, recursively ... ]
//!   [ output switches of this block ]
//! ```
//!
//! `aswaksman_switch_count(n)` equals this layout's length. A later RTL task
//! instantiates switches in exactly this order, so the order is a hard contract.
//!
//! # By-construction agreement
//!
//! [`aswaksman_control`] (control synthesis, via a 2-colouring/looping walk that
//! generalises Beneš's — odd blocks have one pinned unpaired input and one pinned
//! hardwired output) and [`aswaksman_apply`] (fabric simulation) share ONE recursive
//! block decomposition and thread the SAME running control offset, so they agree by
//! construction rather than by two independently-derived wiring formulas. The
//! `random_permutations` round-trip oracle in the tests below proves it — the same
//! discipline `benes.rs` uses.

/// Number of 2×2 switches in an AS-Waksman network on `n` inputs.
///
/// Closed form `ceil(n*log2 n) - n + 1` for `n >= 2`, `0` for `n <= 1`. Computed
/// here via the same recursion the routing uses, so the count and the flat control
/// layout agree by construction: `floor(n/2)` input switches + `ceil(n/2)-1` output
/// switches (`= n-1` total at this level) plus both subnets.
pub fn aswaksman_switch_count(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    (n - 1) + aswaksman_switch_count(n.div_ceil(2)) + aswaksman_switch_count(n / 2)
}

/// Synthesise switch control bits realising `perm` (input `i` routes to output
/// `perm[i]`). Returns a flat `Vec<bool>` of length [`aswaksman_switch_count`]`(perm.len())`
/// in the recursion order documented at the module level.
///
/// Panics if `perm` is not a permutation of `0..perm.len()` (out-of-range or
/// duplicate target), matching the `<=1-per-bank` injectivity the callers rely on.
pub fn aswaksman_control(perm: &[usize]) -> Vec<bool> {
    let n = perm.len();
    // Validate a genuine bijection on 0..n before routing.
    let mut seen = vec![false; n];
    for &o in perm {
        assert!(o < n, "AS-Waksman target {o} out of range {n}");
        assert!(
            !seen[o],
            "duplicate AS-Waksman target {o} (violates injectivity)"
        );
        seen[o] = true;
    }
    let mut ctrl = vec![false; aswaksman_switch_count(n)];
    route(perm, &mut ctrl, 0);
    ctrl
}

/// Simulate the fabric with `ctrl` on `input` (length `N`). Returns `output[o]` =
/// the value routed to output `o`. Test / gen-time guard oracle — not on the RTL
/// hot path. Mirrors [`aswaksman_control`]'s recursion exactly.
pub fn aswaksman_apply(ctrl: &[bool], input: &[usize]) -> Vec<usize> {
    apply_block(ctrl, input, 0).0
}

/// Recursive control-bit synthesis. `perm` is a bijection on `0..n` (the block size);
/// `base` is this block's first switch index in the flat `ctrl`. Returns the offset
/// one past this block's switches (`base + aswaksman_switch_count(n)`), so the caller
/// can lay out sibling blocks without a separate count pass.
fn route(perm: &[usize], ctrl: &mut [bool], base: usize) -> usize {
    let n = perm.len();
    if n <= 1 {
        return base;
    }
    let in_count = n / 2; // floor(n/2) input switches
    let out_count = n.div_ceil(2) - 1; // ceil(n/2)-1 output switches
    let m_up = n.div_ceil(2); // upper subnet size
    let m_lo = n / 2; // lower subnet size
    let paired = 2 * in_count; // paired inputs are 0..paired
    let has_bypass = n % 2 == 1; // odd n => input `paired` (= n-1) bypasses to upper
    let sw_out = 2 * out_count; // outputs 0..sw_out are switched; the rest hardwired

    // Inverse: inv[o] = input routed to output o.
    let mut inv = vec![0usize; n];
    for (i, &o) in perm.iter().enumerate() {
        inv[o] = i;
    }

    // 2-colour each request `e` (input e -> output perm[e]) with the subnet it takes
    // (true = upper, false = lower). Constraints: the two requests sharing an input
    // switch differ; the two sharing an output switch differ. Pinned requests: the
    // odd-n bypass input is forced upper; a hardwired output forces its request's
    // subnet (upper for output `sw_out`, lower for `sw_out+1`). Seed pinned
    // components from their pin first, then colour any remaining (pin-free) cycles
    // arbitrarily — the AS-Waksman fixed-switch choice makes every component's pins
    // parity-consistent (see module ref).
    let mut color: Vec<Option<bool>> = vec![None; n];
    for e in 0..n {
        if let Some(c) = pin_of(e, perm, paired, has_bypass, sw_out) {
            match color[e] {
                None => {
                    color[e] = Some(c);
                    propagate(e, &mut color, perm, &inv, paired, sw_out);
                }
                Some(cv) => assert_eq!(cv, c, "AS-Waksman pin conflict at {e}"),
            }
        }
    }
    for e in 0..n {
        if color[e].is_none() {
            color[e] = Some(true);
            propagate(e, &mut color, perm, &inv, paired, sw_out);
        }
    }

    // Build the two sub-permutations and this block's input-stage control bits.
    let mut up = vec![0usize; m_up];
    let mut lo = vec![0usize; m_lo];
    for e in 0..n {
        let is_up = color[e].expect("coloured");
        let in_pos = if has_bypass && e == paired {
            m_up - 1 // bypass input enters upper subnet at its last position
        } else {
            e / 2 // input switch index == subnet input position
        };
        let o = perm[e];
        if is_up {
            // upper request: hardwired output (o == sw_out) -> last upper output pos
            up[in_pos] = if o >= sw_out { m_up - 1 } else { o / 2 };
        } else {
            // lower request: hardwired output (o == sw_out+1) -> last lower output pos
            lo[in_pos] = if o >= sw_out { m_lo - 1 } else { o / 2 };
        }
    }
    // Input switch i: bar (false) sends input 2i to the upper subnet.
    for i in 0..in_count {
        ctrl[base + i] = !color[2 * i].expect("coloured");
    }

    // Recurse: upper subnet, then lower subnet, threading the running offset.
    let after_in = base + in_count;
    let after_up = route(&up, ctrl, after_in);
    let after_lo = route(&lo, ctrl, after_up);

    // Output switch j: bar (false) sends the upper subnet output to output 2j.
    for j in 0..out_count {
        ctrl[after_lo + j] = !color[inv[2 * j]].expect("coloured");
    }
    after_lo + out_count
}

/// The pinned subnet of request `e`, if any (see the colouring comment in `route`).
fn pin_of(
    e: usize,
    perm: &[usize],
    paired: usize,
    has_bypass: bool,
    sw_out: usize,
) -> Option<bool> {
    if has_bypass && e == paired {
        return Some(true); // bypass input -> upper
    }
    let o = perm[e];
    if o >= sw_out {
        // hardwired output: `sw_out` from upper, `sw_out+1` (even n) from lower
        return Some(o == sw_out);
    }
    None
}

/// Flood the constraint component of `start`, forcing each input/output-switch
/// neighbour to the opposite colour (asserting consistency where already coloured).
fn propagate(
    start: usize,
    color: &mut [Option<bool>],
    perm: &[usize],
    inv: &[usize],
    paired: usize,
    sw_out: usize,
) {
    let mut stack = vec![start];
    while let Some(u) = stack.pop() {
        let cu = color[u].expect("propagate from coloured node");
        // Neighbour sharing u's input switch, and neighbour sharing u's output switch.
        let mut neighbours = [None, None];
        if u < paired {
            neighbours[0] = Some(u ^ 1);
        }
        let o = perm[u];
        if o < sw_out {
            neighbours[1] = Some(inv[o ^ 1]);
        }
        for nb in neighbours.into_iter().flatten() {
            match color[nb] {
                None => {
                    color[nb] = Some(!cu);
                    stack.push(nb);
                }
                Some(cv) => assert_eq!(cv, !cu, "AS-Waksman colouring conflict"),
            }
        }
    }
}

/// Mirrors `route`'s recursion exactly (same input/output pin conventions, same
/// running control offset) so it agrees with the synthesised bits by construction.
/// Returns `(output, next_offset)`.
fn apply_block(ctrl: &[bool], input: &[usize], base: usize) -> (Vec<usize>, usize) {
    let n = input.len();
    if n <= 1 {
        return (input.to_vec(), base);
    }
    let in_count = n / 2;
    let out_count = n.div_ceil(2) - 1;
    let m_up = n.div_ceil(2);
    let m_lo = n / 2;
    let has_bypass = n % 2 == 1;
    let sw_out = 2 * out_count;

    // Input stage: split into the upper and lower subnet inputs.
    let mut upper_in = vec![0usize; m_up];
    let mut lower_in = vec![0usize; m_lo];
    for i in 0..in_count {
        if ctrl[base + i] {
            // cross
            lower_in[i] = input[2 * i];
            upper_in[i] = input[2 * i + 1];
        } else {
            // bar
            upper_in[i] = input[2 * i];
            lower_in[i] = input[2 * i + 1];
        }
    }
    if has_bypass {
        upper_in[m_up - 1] = input[n - 1];
    }

    let after_in = base + in_count;
    let (upper_out, after_up) = apply_block(ctrl, &upper_in, after_in);
    let (lower_out, after_lo) = apply_block(ctrl, &lower_in, after_up);

    // Output stage: switched pairs, then the hardwired last output(s).
    let mut out = vec![0usize; n];
    for j in 0..out_count {
        if ctrl[after_lo + j] {
            // cross
            out[2 * j] = lower_out[j];
            out[2 * j + 1] = upper_out[j];
        } else {
            // bar
            out[2 * j] = upper_out[j];
            out[2 * j + 1] = lower_out[j];
        }
    }
    out[sw_out] = upper_out[m_up - 1];
    if !has_bypass {
        out[sw_out + 1] = lower_out[m_lo - 1];
    }

    (out, after_lo + out_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(perm: &[usize]) {
        let ctrl = aswaksman_control(perm);
        assert_eq!(ctrl.len(), aswaksman_switch_count(perm.len()));
        let ident: Vec<usize> = (0..perm.len()).collect();
        let routed = aswaksman_apply(&ctrl, &ident);
        for (i, &o) in perm.iter().enumerate() {
            assert_eq!(
                routed[o], i,
                "perm {perm:?}: output {o} got {} not {i}",
                routed[o]
            );
        }
    }

    #[test]
    fn identity_small() {
        for n in 1..=17 {
            check(&(0..n).collect::<Vec<_>>());
        }
    }
    #[test]
    fn reverse_small() {
        for n in 1..=17 {
            check(&(0..n).rev().collect::<Vec<_>>());
        }
    }
    #[test]
    fn odd_sizes() {
        for &n in &[3usize, 5, 7, 9, 25, 400, 800] {
            // a fixed non-trivial perm: rotate by 1
            let p: Vec<usize> = (0..n).map(|i| (i + 1) % n).collect();
            check(&p);
        }
    }
    #[test]
    fn random_permutations() {
        // deterministic LCG (no rand dep, matches benes.rs style)
        let mut s = 0x2545F4914F6CDD1Du64;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for &n in &[2usize, 3, 5, 8, 13, 25, 64, 400, 512, 800] {
            for _ in 0..200 {
                let mut p: Vec<usize> = (0..n).collect();
                for i in (1..n).rev() {
                    let j = (rng() as usize) % (i + 1);
                    p.swap(i, j);
                }
                check(&p);
            }
        }
    }
    #[test]
    #[should_panic]
    fn duplicate_target_rejected() {
        let _ = aswaksman_control(&[0, 0, 2]);
    }

    #[test]
    fn switch_count_closed_form() {
        // ceil(n*log2 n) - n + 1 for a few n (matches the recursion source of truth).
        assert_eq!(aswaksman_switch_count(1), 0);
        assert_eq!(aswaksman_switch_count(2), 1);
        assert_eq!(aswaksman_switch_count(4), 5);
        assert_eq!(aswaksman_switch_count(8), 17);
        assert_eq!(aswaksman_switch_count(3), 3);
    }
}
