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

use rand_philox::Philox;

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
        // Key = splitmix64(seed); the within-draw block index lands in counter
        // words 0–1 and `draw_id ^ tag` (the Philox stream) in words 2–3.
        Self {
            inner: Philox::from_u64_seed_stream(seed, draw_id ^ tag),
        }
    }

    /// Next pseudo-random 32-bit word in this draw's stream.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }

    /// Fill `dest` with the next `dest.len()` words of this draw's stream:
    /// exactly the words, and the end state, of `dest.len()` calls to
    /// [`next_u32`](Self::next_u32), generated several Philox blocks at a time.
    #[inline]
    pub fn fill_u32(&mut self, dest: &mut [u32]) {
        self.inner.fill_u32(dest);
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
        rand_philox::u32_to_unit_f64(self.next_u32())
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

    // Stream layout: key = the halves of splitmix64(seed), block index in
    // counter words 0–1, `draw_id ^ tag` in words 2–3. A nonzero draw_id pins
    // the XOR itself, and with it the separation of draws and of tags.
    #[test]
    fn draw_id_xor_tag_fills_the_upper_counter_words() {
        let (seed, draw_id) = (42u64, 0x0123_4567_89ab_cdef_u64);
        let k = rand_philox::splitmix64(seed);
        let key = [k as u32, (k >> 32) as u32];
        for tag in [STREAM_TAG_RESAMPLE, STREAM_TAG_SIGNFLIP] {
            let s = draw_id ^ tag;
            let mut r = CommonStatsRng::new_tagged(seed, draw_id, tag);
            for block in 0..2u32 {
                for want in philox::philox4x32_10([block, 0, s as u32, (s >> 32) as u32], key) {
                    assert_eq!(r.next_u32(), want);
                }
            }
        }
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
