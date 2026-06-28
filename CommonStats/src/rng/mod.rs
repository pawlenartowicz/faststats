//! Counter-based randomness source for the whole ecosystem
//! (permutation-friendly.md §3). Draw *k*'s words are a pure function of
//! `(seed, draw_id)` with zero stored state — the property the dependency-graph
//! cache and 1-vs-N-thread bit-identity both ride on.
//!
//! The [`philox`] core is the verbatim mcpower port; [`CommonStatsRng`] is the
//! draw-addressable adaptation (the within-draw position and the draw id are
//! encoded into the Philox counter, not a streaming state); [`CommonStatsRng::bounded`]
//! is the Lemire unbiased `[0, n)` integer the resample index path needs. Float /
//! unit-interval sampling is deferred to P3 with the `dist` suite — the P2 index
//! path is integer-only.
pub mod philox;

use philox::philox4x32_10;

/// Domain-separation tag XOR'd into `draw_id` so the resample stream never
/// collides with the base-data or (later) synthetic-generation streams
/// (permutation-friendly.md §3). The bytes spell `RESAMPLE`.
pub const STREAM_TAG_RESAMPLE: u64 = 0x5245_5341_4D50_4C45;

/// David Stafford's "Mix13" SplitMix64 finalizer (avalanche function). Mixes a
/// raw `u64` seed into the two Philox key words so low-entropy standalone seeds
/// (0, 1, 2, …) still produce well-separated streams. mcpower uses the same
/// finalizer; consumers that already seed from a node hash pass an
/// already-mixed value and pay only this one extra avalanche.
#[inline]
fn splitmix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Draw-addressable Philox RNG: the crate's resampling randomness surface.
///
/// Unlike a streaming PRNG, every word is a pure function of `(seed, draw_id,
/// within-draw position)` with no carried entropy — re-running draw *k* with the
/// same `(seed, draw_id)` reproduces it exactly, independent of how many draws
/// ran before it or on which thread. The key derives from `seed` (mixed); the
/// counter carries `(position_block, draw_id ^ STREAM_TAG_RESAMPLE)`, so distinct
/// draws are independent Philox sub-streams. Yields `u32` words and Lemire
/// unbiased bounded integers; float sampling is P3.
#[derive(Debug, Clone)]
pub struct CommonStatsRng {
    key: [u32; 2],
    draw_lo: u32,
    draw_hi: u32,
    block: u64,     // counter block within this draw; 4 words per block
    buf: [u32; 4],
    buf_pos: usize, // 0..=4; 4 = exhausted, refill on next draw
}

impl CommonStatsRng {
    /// Open the word stream for one resample draw, keyed by `(seed, draw_id)`.
    ///
    /// `seed`: the consumer's run seed (already node-hash-mixed in SDOC; any
    /// `u64` for the standalone path — internally re-mixed). `draw_id`: the draw
    /// index in `0..B`; XOR'd with [`STREAM_TAG_RESAMPLE`] for domain separation.
    pub fn new(seed: u64, draw_id: u64) -> Self {
        let k = splitmix64(seed);
        let eff = draw_id ^ STREAM_TAG_RESAMPLE;
        Self {
            key: [k as u32, (k >> 32) as u32],
            draw_lo: eff as u32,
            draw_hi: (eff >> 32) as u32,
            block: 0,
            buf: [0; 4],
            buf_pos: 4,
        }
    }

    /// Next pseudo-random 32-bit word in this draw's stream. Refills a 4-word
    /// Philox block when the buffer is exhausted; the block index sits in counter
    /// words 0–1, the draw id in words 2–3.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.buf_pos == 4 {
            let b = self.block;
            self.buf = philox4x32_10([b as u32, (b >> 32) as u32, self.draw_lo, self.draw_hi], self.key);
            self.block = self.block.wrapping_add(1);
            self.buf_pos = 0;
        }
        let w = self.buf[self.buf_pos];
        self.buf_pos += 1;
        w
    }

    /// Unbiased uniform integer in `[0, n)` via Lemire's method (no modulo bias,
    /// unlike `floor(uniform * n)`). Draws extra words only in the rare rejection
    /// zone, so it stays integer-only and reproducible. `n` must be ≥ 1; `n == 1`
    /// always returns 0.
    ///
    /// Lemire (2019), "Fast Random Integer Generation in an Interval": form the
    /// 64-bit product `m = word · n`; its high 32 bits are the candidate. Reject
    /// only when the low 32 bits fall below the threshold `t = 2³² mod n`, which
    /// is exactly the set that would otherwise bias the result.
    #[inline]
    pub fn bounded(&mut self, n: u32) -> u32 {
        debug_assert!(n >= 1, "bounded: n must be >= 1");
        let n64 = n as u64;
        let mut m = self.next_u32() as u64 * n64;
        let mut lo = m as u32; // low 32 bits of the product
        if lo < n {
            let t = (1u64 << 32).wrapping_rem(n64) as u32; // 2^32 mod n
            while lo < t {
                m = self.next_u32() as u64 * n64;
                lo = m as u32;
            }
        }
        (m >> 32) as u32
    }
}

#[cfg(feature = "dist")]
impl CommonStatsRng {
    /// Open-interval uniform in `(0, 1)`.
    ///
    /// Returns `(word + 0.5) / 2^32` for a fresh 32-bit Philox word, so the
    /// result is centered in its bucket and never lands exactly on `0` or `1`.
    /// The open interval is required because `ContinuousCdf::quantile(0)` /
    /// `quantile(1)` may be `±∞`, and inverse-CDF sampling feeds this value
    /// straight into `quantile`.
    ///
    /// # Returns
    /// A `f64` strictly inside `(0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u32() as f64 + 0.5) / 4_294_967_296.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Same (seed, draw_id) reproduces the exact word stream, draw after draw —
    // the determinism leg of the P2 oracle (spec §Validation).
    #[test]
    fn same_key_reproduces_words() {
        let mut a = CommonStatsRng::new(42, 7);
        let mut b = CommonStatsRng::new(42, 7);
        for _ in 0..1000 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    // Crossing a 4-word block boundary still reproduces (buffer refill is keyed,
    // not stateful) — guards the block-index counter layout.
    #[test]
    fn reproduces_across_block_boundaries() {
        let words: Vec<u32> = {
            let mut r = CommonStatsRng::new(1, 0);
            (0..37).map(|_| r.next_u32()).collect()
        };
        let mut r2 = CommonStatsRng::new(1, 0);
        for &w in &words {
            assert_eq!(r2.next_u32(), w);
        }
    }

    #[test]
    fn different_draw_ids_diverge() {
        let mut a = CommonStatsRng::new(42, 0);
        let mut b = CommonStatsRng::new(42, 1);
        let mut diff = 0usize;
        for _ in 0..100 {
            if a.next_u32() != b.next_u32() {
                diff += 1;
            }
        }
        assert!(diff > 90, "different draw_ids must give independent streams");
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = CommonStatsRng::new(0, 5);
        let mut b = CommonStatsRng::new(1, 5);
        let mut diff = 0usize;
        for _ in 0..100 {
            if a.next_u32() != b.next_u32() {
                diff += 1;
            }
        }
        assert!(diff > 90, "different seeds must give independent streams");
    }

    #[test]
    fn bounded_one_is_always_zero() {
        let mut r = CommonStatsRng::new(99, 3);
        for _ in 0..1000 {
            assert_eq!(r.bounded(1), 0);
        }
    }

    #[test]
    fn bounded_stays_in_range() {
        let mut r = CommonStatsRng::new(7, 11);
        for &n in &[2u32, 3, 7, 10, 100, 1000] {
            for _ in 0..5000 {
                assert!(r.bounded(n) < n, "bounded({n}) out of range");
            }
        }
    }

    // No modulo bias: over many draws every bucket in [0, n) is hit with roughly
    // equal frequency. A biased floor(u*n) would systematically over-fill the low
    // buckets; the χ²-style spread check catches gross deviation (spec §Validation).
    #[test]
    fn bounded_is_approximately_uniform() {
        let n = 7u32;
        let draws = 700_000usize;
        let mut counts = [0u64; 7];
        let mut r = CommonStatsRng::new(2024, 1);
        for _ in 0..draws {
            counts[r.bounded(n) as usize] += 1;
        }
        let expected = draws as f64 / n as f64;
        for (i, &c) in counts.iter().enumerate() {
            let rel = (c as f64 - expected).abs() / expected;
            assert!(rel < 0.02, "bucket {i} count {c} deviates {rel:.4} from uniform");
        }
    }

    #[cfg(feature = "dist")]
    #[test]
    fn uniform_in_open_unit_interval() {
        let mut rng = CommonStatsRng::new(42, 0);
        for _ in 0..100_000 {
            let u = rng.uniform();
            assert!(u > 0.0 && u < 1.0, "uniform out of (0,1): {u}");
        }
        // Mean of a large sample is ~0.5 (sanity; not a distribution test).
        let mut rng = CommonStatsRng::new(7, 1);
        let n = 200_000;
        let mean: f64 = (0..n).map(|_| rng.uniform()).sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 1e-2, "mean {mean} far from 0.5");
    }
}
