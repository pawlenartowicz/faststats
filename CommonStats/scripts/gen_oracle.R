#!/usr/bin/env Rscript
# Layer-2 cross-check engine: the scipy+R recompute leg of the three-source gate.
# Reads a job manifest written by gen_oracle.py, computes each quantity once with
# base R `stats`, writes the values back as JSON. One batch invocation per run —
# never called per value, and never a test-time dependency.
#
#   Rscript gen_oracle.R <manifest.json> <out.json>
#
# Manifest: a JSON array of {"func": <name>, "args": [<f64>...]}. Output: a JSON
# array of values in the same order (non-finite / errored → null).
#
# Convention note: every `func` here must mirror the matched parameterization in
# gen_oracle.py's identity map. erf/erfc/erfinv/erfcinv have no base-R primitive,
# so they go through the pnorm/qnorm identities (R-availability rule):
#   erf(x)    = 2*pnorm(x*sqrt2) - 1        erfinv(p)  = qnorm((p+1)/2)/sqrt2
#   erfc(x)   = 2*pnorm(-x*sqrt2)           erfcinv(p) = qnorm(1 - p/2)/sqrt2

SQRT2 <- sqrt(2)

dispatch <- function(func, a) {
  switch(func,
    "erf"        = 2 * pnorm(a[1] * SQRT2) - 1,
    "erfc"       = 2 * pnorm(-a[1] * SQRT2),
    "erfinv"     = qnorm((a[1] + 1) / 2) / SQRT2,
    "erfcinv"    = qnorm(1 - a[1] / 2) / SQRT2,
    "gamma"      = gamma(a[1]),
    "lgamma"     = lgamma(a[1]),
    "digamma"    = digamma(a[1]),
    "lbeta"      = lbeta(a[1], a[2]),
    # logsumexp: stable log-sum-exp over all args (no separate base R fn; manual stable).
    "logsumexp"  = { m <- max(a); if (!is.finite(m)) NA_real_ else m + log(sum(exp(a - m))) },
    # gammp/gammq are the regularized lower/upper incomplete gamma P(a,x)/Q(a,x);
    # pgamma(x, shape=a, rate=1) == P(a,x).
    "gammp"      = pgamma(a[2], shape = a[1]),
    "gammq"      = pgamma(a[2], shape = a[1], lower.tail = FALSE),
    "betai"      = pbeta(a[3], a[1], a[2]),
    "invbetareg" = qbeta(a[3], a[1], a[2]),

    # --- distribution suite (Phase 3). For continuous d/p/q the manifest args are
    # [x_or_p, *params]; for discrete d/p they are [k, *params]. Each func mirrors
    # the matched scipy/mpmath parameterization in gen_oracle.py's gen_dist().
    "dnorm"    = dnorm(a[1], a[2], a[3]),
    "pnorm"    = pnorm(a[1], a[2], a[3]),
    "dt"       = dt(a[1], a[2]),
    "pt"       = pt(a[1], a[2]),
    "dchisq"   = dchisq(a[1], a[2]),
    "pchisq"   = pchisq(a[1], a[2]),
    "df"       = df(a[1], a[2], a[3]),
    "pf"       = pf(a[1], a[2], a[3]),
    "dunif"    = dunif(a[1], a[2], a[3]),
    "punif"    = punif(a[1], a[2], a[3]),
    "dexp"     = dexp(a[1], rate = a[2]),
    "pexp"     = pexp(a[1], rate = a[2]),
    "dcauchy"  = dcauchy(a[1], a[2], a[3]),
    "pcauchy"  = pcauchy(a[1], a[2], a[3]),
    "dweibull" = dweibull(a[1], shape = a[2], scale = a[3]),
    "pweibull" = pweibull(a[1], shape = a[2], scale = a[3]),
    "dlnorm"   = dlnorm(a[1], a[2], a[3]),
    "plnorm"   = plnorm(a[1], a[2], a[3]),
    "dgamma"   = dgamma(a[1], shape = a[2], rate = a[3]),
    "pgamma_d" = pgamma(a[1], shape = a[2], rate = a[3]),
    "dbeta"    = dbeta(a[1], a[2], a[3]),
    "pbeta_d"  = pbeta(a[1], a[2], a[3]),
    "dbinom"   = dbinom(a[1], size = a[2], prob = a[3]),
    "pbinom"   = pbinom(a[1], size = a[2], prob = a[3]),
    "dpois"    = dpois(a[1], lambda = a[2]),
    "ppois"    = ppois(a[1], lambda = a[2]),
    "dgeom"    = dgeom(a[1], prob = a[2]),
    "pgeom"    = pgeom(a[1], prob = a[2]),
    "dnbinom"  = dnbinom(a[1], size = a[2], prob = a[3]),
    "pnbinom"  = pnbinom(a[1], size = a[2], prob = a[3]),
    "dhyper"   = dhyper(a[1], m = a[2], n = a[3], k = a[4]),
    "phyper"   = phyper(a[1], m = a[2], n = a[3], k = a[4]),

    # --- descriptives area. Each unary function receives the data vector as all args.
    # Binary functions (cov/pearson) receive [n, x1…xn, y1…yn] and split at index n.
    # R has no built-in skewness/kurtosis; those two are scipy-only (R n/a).
    "desc_mean"    = mean(a),
    "desc_var"     = var(a),     # sample, ddof=1
    "desc_sd"      = sd(a),      # sample, ddof=1
    "desc_median"  = median(a),
    "desc_cov"     = { n <- as.integer(a[1]); cov(a[2:(n+1)], a[(n+2):(2*n+1)]) },
    "desc_pearson" = { n <- as.integer(a[1]); cor(a[2:(n+1)], a[(n+2):(2*n+1)]) },

    # --- transform area (Phase 4 three-source migration) ---
    # qnorm: standard-normal quantile function (for normal_scores cross-check).
    "qnorm"    = qnorm(a[1]),

    # rank_avg/min/max: base R rank() for scalar cross-check. Args carry the whole
    # data vector plus a trailing 0-based target index, since _run_r returns one
    # scalar per job. R base has no dense/ordinal — those are R n/a in gen_oracle.py.
    "rank_avg" = rank(a[-length(a)], ties.method = "average")[a[length(a)] + 1],
    "rank_min" = rank(a[-length(a)], ties.method = "min")[a[length(a)] + 1],
    "rank_max" = rank(a[-length(a)], ties.method = "max")[a[length(a)] + 1],

    # --- density area (Phase 4 three-source migration) ---
    # kde_manual: exact Gaussian KDE sum (NOT density(), which is FFT-binned).
    # Args: [x_eval, h, data_1, ..., data_n]. One job per grid point.
    "kde_manual" = {
      x_eval <- a[1]; h <- a[2]; data_pts <- a[-(1:2)]; n <- length(data_pts)
      sum(dnorm((x_eval - data_pts) / h)) / (n * h)
    },

    # --- htest area (Phase 5 three-source migration). Vector payloads are
    # length-tagged because _run_r returns one scalar per job; one job per
    # component. Two-sample layout: [na, nb, x..., y...]. The welch conf.int
    # cross-check is what surfaces the known pooled-vs-Welch CI inconsistency.
    "ttest_one_t"      = { mu<-a[1]; x<-a[-1]; unname(t.test(x, mu=mu)$statistic) },
    "ttest_one_p"      = { mu<-a[1]; x<-a[-1]; t.test(x, mu=mu)$p.value },
    "ttest_student_t"  = { na<-a[1]; nb<-a[2]; x<-a[3:(2+na)]; y<-a[(3+na):(2+na+nb)]; unname(t.test(x,y,var.equal=TRUE)$statistic) },
    "ttest_student_p"  = { na<-a[1]; nb<-a[2]; x<-a[3:(2+na)]; y<-a[(3+na):(2+na+nb)]; t.test(x,y,var.equal=TRUE)$p.value },
    "ttest_welch_t"    = { na<-a[1]; nb<-a[2]; x<-a[3:(2+na)]; y<-a[(3+na):(2+na+nb)]; unname(t.test(x,y,var.equal=FALSE)$statistic) },
    "ttest_welch_p"    = { na<-a[1]; nb<-a[2]; x<-a[3:(2+na)]; y<-a[(3+na):(2+na+nb)]; t.test(x,y,var.equal=FALSE)$p.value },
    "ttest_welch_df"   = { na<-a[1]; nb<-a[2]; x<-a[3:(2+na)]; y<-a[(3+na):(2+na+nb)]; unname(t.test(x,y,var.equal=FALSE)$parameter) },
    "ttest_welch_cilo" = { na<-a[1]; nb<-a[2]; x<-a[3:(2+na)]; y<-a[(3+na):(2+na+nb)]; t.test(x,y,var.equal=FALSE)$conf.int[1] },
    "ttest_welch_cihi" = { na<-a[1]; nb<-a[2]; x<-a[3:(2+na)]; y<-a[(3+na):(2+na+nb)]; t.test(x,y,var.equal=FALSE)$conf.int[2] },
    "ttest_paired_t"   = { na<-a[1]; nb<-a[2]; x<-a[3:(2+na)]; y<-a[(3+na):(2+na+nb)]; unname(t.test(x,y,paired=TRUE)$statistic) },
    "ttest_paired_p"   = { na<-a[1]; nb<-a[2]; x<-a[3:(2+na)]; y<-a[(3+na):(2+na+nb)]; t.test(x,y,paired=TRUE)$p.value },
    "anova_f"          = { ng<-a[1]; sizes<-a[2:(1+ng)]; vals<-a[(2+ng):length(a)]; g<-factor(rep(seq_len(ng), sizes)); unname(oneway.test(vals~g, var.equal=TRUE)$statistic) },
    "anova_p"          = { ng<-a[1]; sizes<-a[2:(1+ng)]; vals<-a[(2+ng):length(a)]; g<-factor(rep(seq_len(ng), sizes)); oneway.test(vals~g, var.equal=TRUE)$p.value },
    "chisq_gof_chi2"   = { k<-a[1]; obs<-a[2:(1+k)]; exp<-a[(2+k):(1+2*k)]; unname(chisq.test(obs, p=exp/sum(exp))$statistic) },
    "chisq_gof_p"      = { k<-a[1]; obs<-a[2:(1+k)]; exp<-a[(2+k):(1+2*k)]; chisq.test(obs, p=exp/sum(exp))$p.value },
    "chisq_ind_chi2"   = { nr<-a[1]; nc<-a[2]; m<-matrix(a[3:length(a)], nrow=nr, byrow=TRUE); unname(chisq.test(m, correct=FALSE)$statistic) },
    "chisq_ind_p"      = { nr<-a[1]; nc<-a[2]; m<-matrix(a[3:length(a)], nrow=nr, byrow=TRUE); chisq.test(m, correct=FALSE)$p.value },
    "cor_r"            = { n<-a[1]; x<-a[2:(1+n)]; y<-a[(2+n):(1+2*n)]; unname(cor.test(x,y,method="pearson")$estimate) },
    "cor_t"            = { n<-a[1]; x<-a[2:(1+n)]; y<-a[(2+n):(1+2*n)]; unname(cor.test(x,y,method="pearson")$statistic) },
    "cor_p"            = { n<-a[1]; x<-a[2:(1+n)]; y<-a[(2+n):(1+2*n)]; cor.test(x,y,method="pearson")$p.value },
    "vartest_f"        = { na<-a[1]; nb<-a[2]; x<-a[3:(2+na)]; y<-a[(3+na):(2+na+nb)]; unname(var.test(x,y)$statistic) },
    "vartest_p"        = { na<-a[1]; nb<-a[2]; x<-a[3:(2+na)]; y<-a[(3+na):(2+na+nb)]; var.test(x,y)$p.value },
    stop(paste("gen_oracle.R: unknown func", func))
  )
}

argv <- commandArgs(trailingOnly = TRUE)
if (length(argv) != 2) stop("usage: gen_oracle.R <manifest.json> <out.json>")
manifest_path <- argv[1]; out_path <- argv[2]

# simplifyVector=FALSE keeps each job a plain list (no data.frame coercion of the
# ragged args column), so 1/2/3-arg quantities coexist in one manifest.
jobs <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)

vals <- vapply(jobs, function(j) {
  v <- tryCatch(dispatch(j$func, as.numeric(unlist(j$args))),
                error = function(e) NA_real_)
  if (length(v) != 1 || !is.finite(v)) NA_real_ else as.numeric(v)
}, numeric(1))

# digits=NA → full 17-sig-digit round-trippable output; NA → JSON null.
writeLines(jsonlite::toJSON(vals, digits = NA, na = "null"), out_path)
cat(sprintf("gen_oracle.R: wrote %d values to %s\n", length(vals), out_path))
