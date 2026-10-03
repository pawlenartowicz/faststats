//! Evaluates a crate function by the name the accuracy harness
//! (`scripts/accuracy_sweep.py`) gives it, at exact inputs. Shared by
//! `scripts/accuracy_probe.rs` (the sweep's evaluator) and
//! `tests/accuracy.rs` (the replay of exported points), so both read the same
//! names.
//!
//! Names: the special functions by their own name (`gammp`, `gammq`,
//! `ln_gammp`, `ln_gammq`, `betai`, `ln_betai`, `lgamma`, `lbeta`,
//! `inv_beta_reg`), distribution methods as `<type>.<method>` with the type in
//! lower case (`gamma.cdf`, `negbinomial_ms.sf` for
//! `NegBinomial::from_mean_size`). `params` are the constructor's arguments
//! (the special functions' leading arguments), `x` the method's argument.
#![allow(dead_code)]

#[cfg(feature = "dist")]
use commonstats::dist::{
    Bernoulli, Beta, Binomial, Cauchy, ChiSquared, ContinuousCdf, ContinuousDensity, DiscreteCdf,
    DiscreteMass, Exponential, FisherF, Gamma, Geometric, Hypergeometric, InverseGaussian,
    LogNormal, NegBinomial, Normal, Poisson, StudentT, Uniform, Weibull,
};
use commonstats::special;

/// One input, written as text: digits only is an exact `i64` (integer
/// parameters above 2⁵³ stay exact), anything else (a `.`, an exponent, `inf`,
/// `nan`) an `f64`, parsed with correct rounding.
#[derive(Clone, Copy, Debug)]
pub enum Arg {
    F(f64),
    I(i64),
}

impl Arg {
    pub fn parse(s: &str) -> Result<Arg, String> {
        if let Ok(i) = s.parse::<i64>() {
            return Ok(Arg::I(i));
        }
        s.parse::<f64>()
            .map(Arg::F)
            .map_err(|e| format!("bad number {s:?}: {e}"))
    }
    pub fn f(self) -> f64 {
        match self {
            Arg::F(v) => v,
            Arg::I(i) => i as f64,
        }
    }
    pub fn i(self) -> Result<i64, String> {
        match self {
            Arg::I(i) => Ok(i),
            Arg::F(v) => Err(format!("integer expected, got {v:?}")),
        }
    }
}

/// A result: an `f64`, or an `i64` for the discrete quantiles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Val {
    F(f64),
    I(i64),
}

fn param(params: &[Arg], i: usize) -> Result<Arg, String> {
    params
        .get(i)
        .copied()
        .ok_or_else(|| format!("missing parameter {i}"))
}

pub fn eval(name: &str, params: &[Arg], x: Arg) -> Result<Val, String> {
    let p = |i: usize| param(params, i).map(Arg::f);
    let v = match name {
        "gammp" => special::gammp(p(0)?, x.f()),
        "gammq" => special::gammq(p(0)?, x.f()),
        "ln_gammp" => special::incomplete::ln_gammp(p(0)?, x.f()),
        "ln_gammq" => special::incomplete::ln_gammq(p(0)?, x.f()),
        "betai" => special::betai(p(0)?, p(1)?, x.f()),
        // `y = 1 − x` as `betai` forms it.
        "ln_betai" => special::incomplete::ln_betai(p(0)?, p(1)?, x.f(), 1.0 - x.f()),
        "lgamma" => special::lgamma(x.f()),
        "lbeta" => special::lbeta(p(0)?, x.f()),
        "inv_beta_reg" => special::inv_beta_reg(p(0)?, p(1)?, x.f()),
        _ => return eval_dist(name, params, x),
    };
    Ok(Val::F(v))
}

#[cfg(not(feature = "dist"))]
fn eval_dist(name: &str, _params: &[Arg], _x: Arg) -> Result<Val, String> {
    Err(format!("{name}: needs feature `dist`"))
}

#[cfg(feature = "dist")]
fn eval_dist(name: &str, params: &[Arg], x: Arg) -> Result<Val, String> {
    let (ty, method) = name
        .split_once('.')
        .ok_or_else(|| format!("unknown function {name}"))?;
    let p = |i: usize| param(params, i).map(Arg::f);
    let e = |e: commonstats::StatError| format!("{name}: {e}");
    match (ty, method) {
        ("fisherf", "ln_cdf") => {
            return Ok(Val::F(FisherF::new(p(0)?, p(1)?).map_err(e)?.ln_cdf(x.f())));
        }
        ("fisherf", "ln_sf") => {
            return Ok(Val::F(FisherF::new(p(0)?, p(1)?).map_err(e)?.ln_sf(x.f())));
        }
        ("gamma", "ln_cdf") => {
            return Ok(Val::F(Gamma::new(p(0)?, p(1)?).map_err(e)?.ln_cdf(x.f())));
        }
        ("inversegaussian", "ln_cdf") => {
            return Ok(Val::F(
                InverseGaussian::new(p(0)?, p(1)?).map_err(e)?.ln_cdf(x.f()),
            ));
        }
        ("inversegaussian", "ln_sf") => {
            return Ok(Val::F(
                InverseGaussian::new(p(0)?, p(1)?).map_err(e)?.ln_sf(x.f()),
            ));
        }
        _ => {}
    }
    match ty {
        "normal" => cont(Normal::new(p(0)?, p(1)?).map_err(e)?, method, x),
        "studentt" => cont(StudentT::new(p(0)?).map_err(e)?, method, x),
        "chisquared" => cont(ChiSquared::new(p(0)?).map_err(e)?, method, x),
        "fisherf" => cont(FisherF::new(p(0)?, p(1)?).map_err(e)?, method, x),
        "uniform" => cont(Uniform::new(p(0)?, p(1)?).map_err(e)?, method, x),
        "exponential" => cont(Exponential::new(p(0)?).map_err(e)?, method, x),
        "cauchy" => cont(Cauchy::new(p(0)?, p(1)?).map_err(e)?, method, x),
        "weibull" => cont(Weibull::new(p(0)?, p(1)?).map_err(e)?, method, x),
        "lognormal" => cont(LogNormal::new(p(0)?, p(1)?).map_err(e)?, method, x),
        "gamma" => cont(Gamma::new(p(0)?, p(1)?).map_err(e)?, method, x),
        "beta" => cont(Beta::new(p(0)?, p(1)?).map_err(e)?, method, x),
        "inversegaussian" => cont(InverseGaussian::new(p(0)?, p(1)?).map_err(e)?, method, x),
        "bernoulli" => disc(Bernoulli::new(p(0)?).map_err(e)?, method, x),
        "binomial" => disc(
            Binomial::new(param(params, 0)?.i()?, p(1)?).map_err(e)?,
            method,
            x,
        ),
        "poisson" => disc(Poisson::new(p(0)?).map_err(e)?, method, x),
        "geometric" => disc(Geometric::new(p(0)?).map_err(e)?, method, x),
        "negbinomial" => disc(NegBinomial::new(p(0)?, p(1)?).map_err(e)?, method, x),
        "negbinomial_ms" => disc(
            NegBinomial::from_mean_size(p(0)?, p(1)?).map_err(e)?,
            method,
            x,
        ),
        "hypergeometric" => disc(
            Hypergeometric::new(
                param(params, 0)?.i()?,
                param(params, 1)?.i()?,
                param(params, 2)?.i()?,
            )
            .map_err(e)?,
            method,
            x,
        ),
        _ => Err(format!("unknown function {name}")),
    }
}

#[cfg(feature = "dist")]
fn cont<D: ContinuousCdf + ContinuousDensity>(d: D, method: &str, x: Arg) -> Result<Val, String> {
    let x = x.f();
    let e = |e: commonstats::StatError| format!("{method}: {e}");
    Ok(Val::F(match method {
        "cdf" => d.cdf(x),
        "sf" => d.sf(x),
        "pdf" => d.density(x),
        "log_density" => d.log_density(x),
        "quantile" => d.quantile(x).map_err(e)?,
        "isf" => d.isf(x).map_err(e)?,
        _ => return Err(format!("unknown method {method}")),
    }))
}

#[cfg(feature = "dist")]
fn disc<D: DiscreteCdf + DiscreteMass>(d: D, method: &str, x: Arg) -> Result<Val, String> {
    let e = |e: commonstats::StatError| format!("{method}: {e}");
    Ok(match method {
        "mass" => Val::F(d.mass(x.i()?)),
        "log_mass" => Val::F(d.log_mass(x.i()?)),
        "cdf" => Val::F(d.cdf(x.i()?)),
        "sf" => Val::F(d.sf(x.i()?)),
        "quantile" => Val::I(d.quantile(x.f()).map_err(e)?),
        _ => return Err(format!("unknown method {method}")),
    })
}
