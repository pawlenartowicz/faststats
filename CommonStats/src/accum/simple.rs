//! Trivially-mergeable accumulators: Count, Sum (Neumaier), MinMax.
use super::{Accumulator, Mergeable};

/// Running count of observations folded in (NaN filtered upstream by `from_slice`).
#[derive(Clone, Default)]
pub struct Count {
    n: u64,
}
impl Mergeable for Count {
    fn merge(&mut self, o: &Self) {
        self.n += o.n;
    }
}
impl Accumulator for Count {
    type Item = f64;
    type Output = u64;
    fn empty() -> Self {
        Self::default()
    }
    fn update(&mut self, _x: f64) {
        self.n += 1;
    }
    fn finalize(&self) -> u64 {
        self.n
    }
}

/// Neumaier-compensated sum (Kahan variant robust to large-then-small ordering).
#[derive(Clone, Default)]
pub struct Sum {
    sum: f64,
    c: f64,
}
impl Mergeable for Sum {
    // Feed the other's running sum through the compensated add, then carry its
    // leftover correction across; true total stays sum + c on both sides.
    fn merge(&mut self, o: &Self) {
        self.add(o.sum);
        self.c += o.c;
    }
}
impl Sum {
    fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.c += (self.sum - t) + x;
        } else {
            self.c += (x - t) + self.sum;
        }
        self.sum = t;
    }
}
impl Accumulator for Sum {
    type Item = f64;
    type Output = f64;
    fn empty() -> Self {
        Self::default()
    }
    fn update(&mut self, x: f64) {
        self.add(x);
    }
    fn finalize(&self) -> f64 {
        self.sum + self.c
    }
}

/// Running minimum and maximum; `finalize` is `None` until the first observation.
#[derive(Clone)]
pub struct MinMax {
    lo: f64,
    hi: f64,
    seen: bool,
}
impl Default for MinMax {
    fn default() -> Self {
        Self {
            lo: f64::INFINITY,
            hi: f64::NEG_INFINITY,
            seen: false,
        }
    }
}
impl Mergeable for MinMax {
    fn merge(&mut self, o: &Self) {
        if o.seen {
            self.lo = self.lo.min(o.lo);
            self.hi = self.hi.max(o.hi);
            self.seen = true;
        }
    }
}
impl Accumulator for MinMax {
    type Item = f64;
    type Output = Option<(f64, f64)>;
    fn empty() -> Self {
        Self::default()
    }
    fn update(&mut self, x: f64) {
        self.lo = self.lo.min(x);
        self.hi = self.hi.max(x);
        self.seen = true;
    }
    fn finalize(&self) -> Option<(f64, f64)> {
        if self.seen {
            Some((self.lo, self.hi))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accum::Accumulator;

    #[test]
    fn count_skips_nan() {
        let c: Count = crate::accum::from_slice(&[1.0, f64::NAN, 3.0]);
        assert_eq!(c.finalize(), 2);
    }
    #[test]
    fn sum_is_compensated() {
        // 1e16 + 1 + -1e16 loses the 1 in naive f64; Neumaier keeps it.
        let s: Sum = crate::accum::from_slice(&[1e16, 1.0, -1e16]);
        assert_eq!(s.finalize(), 1.0);
    }
    #[test]
    fn merge_equals_fold() {
        let a: Sum = crate::accum::from_slice(&[1.0, 2.0]);
        let b: Sum = crate::accum::from_slice(&[3.0, 4.0]);
        let mut m = a;
        m.merge(&b);
        assert_eq!(m.finalize(), 10.0);
    }
    #[test]
    fn minmax_empty_is_none() {
        let mm: MinMax = crate::accum::from_slice(&[f64::NAN]);
        assert_eq!(mm.finalize(), None);
    }
    #[test]
    fn minmax_range() {
        let mm: MinMax = crate::accum::from_slice(&[3.0, 1.0, 2.0]);
        assert_eq!(mm.finalize(), Some((1.0, 3.0)));
    }
}
