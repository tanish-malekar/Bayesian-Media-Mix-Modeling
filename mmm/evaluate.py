import numpy as np
import pandas as pd


def metrics(actual, prediction, lower=None, upper=None):
    actual, prediction = np.asarray(actual, float), np.asarray(prediction, float)
    if actual.shape != prediction.shape or actual.size == 0 or not np.isfinite(actual).all() or not np.isfinite(prediction).all() or np.any(actual <= 0):
        raise ValueError("MAPE requires finite, aligned predictions and positive observations")
    result = {"mape_percent": float(np.mean(np.abs(actual - prediction) / actual) * 100),
              "wmape_percent": float(np.abs(actual - prediction).sum() / actual.sum() * 100),
              "rmse": float(np.sqrt(np.mean((actual - prediction) ** 2)))}
    if lower is not None and upper is not None:
        result["predictive_interval_90_coverage"] = float(np.mean((actual >= lower) & (actual <= upper)))
    return result


def prediction_frame(frame, means, sigma, seed=42):
    rng = np.random.default_rng(seed)
    predictive = means + rng.normal(size=means.shape) * np.asarray(sigma)[:, None]
    return pd.DataFrame({"date": frame.DATE, "actual": frame.revenue,
                         "prediction": means.mean(axis=0),
                         "mean_p05": np.quantile(means, .05, axis=0), "mean_p95": np.quantile(means, .95, axis=0),
                         "predictive_p05": np.quantile(predictive, .05, axis=0), "predictive_p95": np.quantile(predictive, .95, axis=0)})
