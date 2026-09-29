//! Counter-based randomness source for the whole ecosystem.
//! Draw *k*'s words are a pure function of
//! `(seed, draw_id)` with zero stored state — the property the dependency-graph
//! cache and 1-vs-N-thread bit-identity both ride on.
//!
//! The Philox4x32-10 counter-based PRNG (Random123) comes from `rand_philox`;
//! [`CommonStatsRng`] is the draw-addressable adaptation (the within-draw
//! position and the draw id are encoded into the Philox counter, not a streaming
//! state); [`CommonStatsRng::bounded`] is the Lemire unbiased `[0, n)` integer the
//! resample index path needs.
//! Float/unit-interval sampling is deferred to the `dist` feature — this module
//! (integer-only path) generates indices and bounded integers.

/// The Philox4x32-10 block function, re-exported from `rand_philox`.
pub mod philox {
    pub use rand_philox::philox4x32_10;
}

use rand_philox::{Philox, splitmix64};

/// Domain-separation tag XOR'd into `draw_id` so the resample stream never
/// collides with the other tagged streams (sign flips, the RLRT null,
/// simulated responses).
/// The bytes spell `RESAMPLE`.
pub const STREAM_TAG_RESAMPLE: u64 = 0x5245_5341_4D50_4C45;

/// Domain-separation tag for the sign-flip stream ([`gen_sign_flips`]) so sign
/// flips and row permutations of the same `draw_id` never share random bits.
/// The bytes spell `SIGNFLIP`.
///
/// [`gen_sign_flips`]: crate::resample::gen_sign_flips
pub const STREAM_TAG_SIGNFLIP: u64 = 0x5349_474E_464C_4950;

/// Domain-separation tag for the simulated null distribution of the exact
/// restricted likelihood-ratio test, so its draws never share random bits with
/// the resample or sign-flip streams of the same `draw_id`. The bytes spell
/// `RLRTNULL`.
pub const STREAM_TAG_RLRT: u64 = 0x524C_5254_4E55_4C4C;

/// Domain-separation tag for simulated responses (draws from a fitted or
/// assumed model), so they never share random bits with the resampling indices
/// of the same `(seed, draw_id)`. The bytes spell `SIMULATE`.
pub const STREAM_TAG_SIMULATE: u64 = 0x5349_4D55_4C41_5445;

/// Draw-addressable Philox RNG: the crate's resampling randomness surface.
///
/// Unlike a streaming PRNG, every word is a pure function of `(seed, draw_id,
/// within-draw position)` with no carried entropy — re-running draw *k* with the
/// same `(seed, draw_id)` reproduces it exactly, independent of how many draws
/// ran before it or on which thread. The key is `splitmix64(seed)`, so
/// low-entropy standalone seeds (0, 1, 2, …) still give well-separated streams;
/// the counter carries `(position_block, draw_id ^ tag)`, tag
/// `STREAM_TAG_RESAMPLE` by default, so distinct draws are independent Philox
/// sub-streams. Yields `u32` words and Lemire unbiased bounded integers; with
/// the `dist` feature, also open-interval uniform floats in `(0, 1)`
/// (`uniform`, `uniform52`).
#[derive(Debug, Clone)]
pub struct CommonStatsRng {
    inner: Philox,
}

impl CommonStatsRng {
    /// Open the word stream for one resample draw, keyed by `(seed, draw_id)`.
    ///
    /// `seed`: the consumer's run seed (already node-hash-mixed in SDOC; any
    /// `u64` for the standalone path — internally re-mixed). `draw_id`: the draw
    /// index in `0..B`; XOR'd with [`STREAM_TAG_RESAMPLE`] for domain separation.
    /// Equals `new_tagged(seed, draw_id, STREAM_TAG_RESAMPLE)`.
    pub fn new(seed: u64, draw_id: u64) -> Self {
        Self::new_tagged(seed, draw_id, STREAM_TAG_RESAMPLE)
    }

    /// [`new`](Self::new) with an explicit domain-separation `tag` XOR'd into
    /// `draw_id` — one of the `STREAM_TAG_*` constants. Streams with different
    /// tags and the same `(seed, draw_id)` are independent Philox sub-streams.
    pub fn new_tagged(seed: u64, draw_id: u64, tag: u64) -> Self {
        let k = splitmix64(seed);
        // Philox splits the u128 counter into its four words little-endian: the
        // within-draw block index lands in words 0–1, `draw_id ^ tag` in words 2–3.
        let counter = u128::from(draw_id ^ tag) << 64;
        Self {
            inner: Philox::new([k as u32, (k >> 32) as u32], counter),
        }
    }

    /// Next pseudo-random 32-bit word in this draw's stream.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }

    /// Unbiased uniform integer in `[0, n)` via Lemire's method (no modulo bias,
    /// unlike `floor(uniform * n)`). Draws extra words only in the rare rejection
    /// zone, so it stays integer-only and reproducible. `n` must be ≥ 1; `n == 1`
    /// always returns 0.
    ///
    /// Lemire (2019), "Fast Random Integer Generation in an Interval".
    #[inline]
    pub fn bounded(&mut self, n: u32) -> u32 {
        self.inner.bounded(n)
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

    /// Open-interval uniform in `(0, 1)` with 52 random bits, for samplers
    /// whose output resolves finer than `uniform`'s 2⁻³² grid.
    ///
    /// Returns `(x + 0.5) / 2^52` for `x` the 32 bits of one fresh Philox word
    /// followed by the top 20 bits of the next: the odd multiples of 2⁻⁵³,
    /// each exact in f64, so the result lies in `[2⁻⁵³, 1 − 2⁻⁵³]`. Consumes
    /// two words.
    ///
    /// # Returns
    /// A `f64` strictly inside `(0, 1)`.
    pub fn uniform52(&mut self) -> f64 {
        let hi = self.next_u32() as u64;
        let lo = self.next_u32() as u64;
        let x = (hi << 20) | (lo >> 12);
        (x as f64 + 0.5) / 4_503_599_627_370_496.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    // Same (seed, draw_id) reproduces the exact word stream, draw after draw —
    // the determinism guarantee the resampling layer relies on.
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
        assert!(
            diff > 90,
            "different draw_ids must give independent streams"
        );
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

    // `new` is `new_tagged` with the resample tag — existing fixtures stay byte-identical.
    #[test]
    fn new_equals_new_tagged_resample() {
        let mut a = CommonStatsRng::new(42, 7);
        let mut b = CommonStatsRng::new_tagged(42, 7, STREAM_TAG_RESAMPLE);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn signflip_tag_diverges_from_resample_stream() {
        let mut a = CommonStatsRng::new_tagged(42, 7, STREAM_TAG_RESAMPLE);
        let mut b = CommonStatsRng::new_tagged(42, 7, STREAM_TAG_SIGNFLIP);
        let diff = (0..100).filter(|_| a.next_u32() != b.next_u32()).count();
        assert!(diff > 90, "tags must separate streams");
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
    // buckets; the χ²-style spread check catches gross deviation.
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
            assert!(
                rel < 0.02,
                "bucket {i} count {c} deviates {rel:.4} from uniform"
            );
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

    // `uniform52` is `(x + 0.5)/2⁵²` for `x` = first word ‖ top 20 bits of the
    // second, and takes exactly two words.
    #[cfg(feature = "dist")]
    #[test]
    fn uniform52_bits_and_word_count() {
        let mut words = CommonStatsRng::new(9, 4);
        let mut rng = CommonStatsRng::new(9, 4);
        for _ in 0..10_000 {
            let (w0, w1) = (words.next_u32() as u64, words.next_u32() as u64);
            let x = (w0 << 20) | (w1 >> 12);
            let u = rng.uniform52();
            assert_eq!(u, (x as f64 + 0.5) / 4_503_599_627_370_496.0);
            assert!(u > 0.0 && u < 1.0, "uniform52 out of (0,1): {u}");
        }
        assert_eq!(rng.next_u32(), words.next_u32(), "word count differs");
    }
}
