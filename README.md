# Bayesian Media Mix Modeling

A Python marketing measurement project using PyMC, SciPy, and pandas. It models
channel carryover, diminishing returns, annual seasonality, trend, and observed
business context, then evaluates a chronological holdout and compares constrained
budget scenarios under posterior uncertainty.

## Data

The repository vendors Meta Robyn's simulated weekly demo data: **208 weeks** from
November 2015 to November 2019, with five spend channels: TV, out-of-home, print,
Facebook, and search. Revenue is the target. Competitor sales, newsletter activity,
and event indicators are context controls. The input uses spend rather than
Facebook impressions or search-click exposure variables.

The CSV is pinned to an upstream commit and SHA-256 in `datasets/provenance.json`.
Its upstream MIT license is retained in `datasets/ROByn-LICENSE`. This is synthetic
demo data, not a disclosed advertiser's production history. Attribution and source
links are in `datasets/README.md`.

## Model

For channel c and week t, geometric carryover follows `a[t,c] = x[t,c] +
decay[c] * a[t-1,c]`. A Hill curve maps carryover to a bounded response. Positive
media coefficients multiply these responses; an intercept and regularized context,
trend, and Fourier coefficients complete the revenue mean. The likelihood is
Gaussian on revenue scaled by the training mean.

Decay has a Beta prior, Hill half-saturation and slope have log-normal priors,
media effects have half-normal priors, and context effects have zero-centered
normal priors. PyMC samples the joint posterior with NUTS. The pipeline exports
posterior summaries, R-hat, effective sample sizes, divergence counts, predictions,
and predictive intervals. These diagnostics help assess whether the posterior
supports an interpretation of response curves.

Spend scales, context standardization, event vocabulary, revenue scaling, and trend
centering are learned from the first **182 weeks** only. The last **26 weeks** are
held out chronologically. Carryover crosses the split using preceding spend,
without resetting the adstock state or conditioning on holdout revenue. Evaluation
conditions on observed holdout spend and context; it is a conditional revenue
prediction exercise rather than a forecast of those future inputs.

## Budget allocation

SciPy SLSQP optimizes channel shares under a fixed total weekly budget and
per-channel bounds. The baseline is the last eight training weeks' average spend.
Default bounds allow 50–150% of that baseline for each channel, over a thirteen-
week scenario. Zero-baseline channels remain at zero under these bounds.

Multiple feasible starts reduce dependence on one local solution. All scenarios
share the same draw-specific starting carryover, future calendar, and fixed last-
training context. The allocator reports baseline and proposed spend, model-implied
revenue, posterior uplift intervals, and probability of positive uplift. It uses
a seeded subset of posterior draws for optimization and retains the budget sum.

These are conditional model projections. Observational correlations, response-curve
identification, prior choices, omitted variables, and extrapolation affect their
interpretation; a modeled scenario is distinct from experimentally measured lift.
The dataset's finite spend range and channels' correlated activity should inform
how narrowly allocation bounds are chosen.

## Setup and usage

Use Python 3.11–3.13 from the repository checkout. Numerical dependencies are
pinned in `pyproject.toml`.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
mmm --config configs/default.json --output artifacts/experiment-01
```

The output directory must be new, preserving prior experiments. Sampling defaults
to four chains, 1,000 tuning steps, 1,000 retained draws per chain, and a fixed seed.
Edit the JSON configuration to change sampling, Fourier order, or scenario bounds.
The full pipeline is also available through `python -m mmm.cli`.

```sh
python scripts/fetch_data.py
python -m pytest
```

The fetch command checks the pinned upstream checksum before replacing the CSV.
Test contracts cover carryover across a split, saturation, budget feasibility,
training-only preprocessing, and metric definitions.

## Outputs

- `posterior.nc` and `posterior_summary.csv` — draws and parameter diagnostics
- `preprocessor.json` — training-derived scales and feature definitions
- `predictions.csv` — observed revenue, posterior mean, intervals, and split labels
- `evaluation.json` — holdout MAPE, WMAPE, RMSE, coverage, dates, hash, and settings
- `allocation.json` — fixed-budget channel scenario and posterior revenue comparison
- `holdout.png` — observed and predicted revenue with the holdout boundary

MAPE and WMAPE are percentages; predictive coverage is a fraction. Allocation
uplift is a model-relative percentage over the configured horizon. Generated
experiment outputs live outside versioned source under `artifacts/`.

## Layout

`mmm/data.py` handles validation and training-derived transforms; `model.py`
defines the Bayesian model; `transforms.py` implements carryover and saturation;
`evaluate.py` defines metrics; `allocate.py` constructs and solves budget scenarios;
`cli.py` writes reproducible experiment artifacts.

Sources: [Meta Robyn](https://github.com/facebookexperimental/Robyn),
[PyMC](https://www.pymc.io/), and
[SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html).

Project code is MIT licensed. The vendored demo data retains its upstream license.
