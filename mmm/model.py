from dataclasses import dataclass
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from .data import CHANNELS
from .transforms import adstock, hill, symbolic_adstock


@dataclass
class Sampling:
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    cores: int = 4
    target_accept: float = .95
    seed: int = 42


def fit(training, processor, config):
    if min(config.draws, config.tune) < 100 or config.chains < 2 or config.cores < 1 or not .5 < config.target_accept < 1:
        raise ValueError("Use >=100 draws/tune, >=2 chains, positive cores and target_accept in (.5,1)")
    spend, features = processor.transform(training)
    y = training.revenue.to_numpy(float) / processor.revenue_scale
    with pm.Model(coords={"channel": CHANNELS, "feature": processor.feature_names, "week": np.arange(len(y))}) as model:
        decay = pm.Beta("decay", alpha=2, beta=4, dims="channel")
        half = pm.LogNormal("half_saturation", mu=0, sigma=.6, dims="channel")
        slope = pm.LogNormal("slope", mu=0, sigma=.35, dims="channel")
        beta = pm.HalfNormal("media_beta", sigma=.5, dims="channel")
        gamma = pm.Normal("control_beta", mu=0, sigma=.25, dims="feature")
        intercept = pm.Normal("intercept", mu=1, sigma=.5)
        sigma = pm.HalfNormal("sigma", sigma=.2)
        carryover = symbolic_adstock(spend, decay)
        # Stable Hill response with exact zero response at zero carryover.
        saturated = pt.switch(pt.gt(carryover, 0),
                              pt.sigmoid(slope * (pt.log(pt.maximum(carryover, 1e-12)) - pt.log(half))), 0)
        mu = intercept + pt.dot(features, gamma) + pt.sum(saturated * beta, axis=1)
        pm.Normal("observed_revenue", mu=mu, sigma=sigma, observed=y, dims="week")
        trace = pm.sample(draws=config.draws, tune=config.tune, chains=config.chains, cores=config.cores,
                          target_accept=config.target_accept, random_seed=config.seed,
                          return_inferencedata=True, idata_kwargs={"log_likelihood": True})
    return trace


def posterior_arrays(trace, max_draws=None, seed=42):
    result = {}
    for name in ("decay", "half_saturation", "slope", "media_beta", "control_beta", "intercept", "sigma"):
        value = trace.posterior[name].to_numpy()
        result[name] = value.reshape((-1,) + value.shape[2:])
    count = len(result["intercept"])
    if max_draws and max_draws < count:
        indices = np.sort(np.random.default_rng(seed).choice(count, max_draws, replace=False))
        result = {key: value[indices] for key, value in result.items()}
    return result


def predict_mean_draws(frame, processor, posterior):
    spend, features = processor.transform(frame)
    means = np.empty((len(posterior["intercept"]), len(frame)))
    for draw in range(len(means)):
        media = hill(adstock(spend, posterior["decay"][draw]),
                     posterior["half_saturation"][draw], posterior["slope"][draw])
        means[draw] = (posterior["intercept"][draw] + features @ posterior["control_beta"][draw]
                       + media @ posterior["media_beta"][draw]) * processor.revenue_scale
    return means
