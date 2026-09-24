# FastStats

[![CI](https://github.com/pawlenartowicz/faststats/actions/workflows/ci.yml/badge.svg)](https://github.com/pawlenartowicz/faststats/actions/workflows/ci.yml)
[![commonstats](https://img.shields.io/crates/v/commonstats.svg?label=commonstats)](https://crates.io/crates/commonstats)
[![License](https://img.shields.io/badge/license-LGPL--3.0--or--later-blue.svg)](#license)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22729417.svg)](https://doi.org/10.5281/zenodo.22729417)

FastStats aims at an ecosystem for statistical computing that is fast, and easy
to use and to install. Rust gives the speed and general safety in memory management. 
Leaving out heavy system libraries like BLAS and LAPACK keeps the installation simple.

## In this repository

| Part | What |
|---|---|
| CommonStats | Classical hypothesis tests, distributions, density estimation, resampling |
| NeuroStats | Neuroimaging inference, currently TFCE, other analyses planned |
| RobustStats | Robust alternatives to the popular tests, planned |

## Separate repositories, same ideas

| Project | What |
|---|---|
| [GLMM](https://github.com/pawlenartowicz/glmm) | Mixed models — linear, generalized, and generalized linear mixed models over one Rust kernel |
| [PLSKit](https://github.com/pawlenartowicz/plskit) | Partial least squares regression |

Each ships on its own schedule and is used from here like any other library.

## Planned

- **RobustStats**, in dependency order: Brunner–Munzel, then energy and kernel
  methods, then robust regression.
- **One Python package** covering every engine, so a single install replaces
  three.
- **Group-level neuroimaging inference**, extending the permutation machinery
  from one-sample tests to general linear model contrasts, built on GLMM.
- **Later, possibly**: ridge regression, factor analysis, simple Bayesian
  methods, small neural networks for teaching.

## Documentation

- [CommonStats crate guide](Documentation/CommonStats/rust/README.md) — modules, features, conventions, validation
- [CommonStats for Python](Documentation/CommonStats/python/README.md) — binding status
- [NeuroStats crate guide](Documentation/NeuroStats/rust/README.md) — full API detail
- [NeuroStats for Python](Documentation/NeuroStats/python/api.md) — Python API
- [NeuroStats and nilearn](Documentation/NeuroStats/python/nilearn_compatibility.md) — compatibility notes
- [RobustStats](Documentation/RobustStats/README.md) — status
- [GLMM](Documentation/GLMM/README.md) — pointer to its own repository

## Citation

If you use FastStats in your research, please cite it:

> Lenartowicz, P. (2026). *FastStats — fast-computing ecosystem for statistics*
> (Version v0.1.0) [Computer software]. Zenodo.
> https://doi.org/10.5281/zenodo.22729418

```bibtex
@software{lenartowicz_2026_faststats,
  author    = {Lenartowicz, Paweł},
  title     = {FastStats — fast-computing ecosystem for statistics},
  month     = sep,
  year      = 2026,
  publisher = {Zenodo},
  version   = {v0.1.0},
  doi       = {10.5281/zenodo.22729417},
  url       = {https://doi.org/10.5281/zenodo.22729417}
}
```

## License

[LGPL-3.0-or-later](LICENSE) for everything in this repository. The LGPL is a
set of extra permissions on top of the [GPL-3.0](LICENSE-GPL), so both texts
ship. CommonStats 0.1.0 as published remains under its original MIT OR
Apache-2.0 terms; later releases are LGPL-3.0-or-later.
