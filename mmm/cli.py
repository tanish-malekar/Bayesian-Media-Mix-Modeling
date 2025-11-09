import argparse
from dataclasses import asdict
from pathlib import Path
import json
import arviz as az
import numpy as np
from .allocate import Allocation, optimize_budget
from .data import DEFAULT_DATA, Preprocessor, fingerprint, load_data
from .evaluate import metrics, prediction_frame
from .model import Sampling, fit, posterior_arrays, predict_mean_draws


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Fit and evaluate a Bayesian media mix model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    holdout = int(config.get("holdout_weeks", 26))
    frame = load_data(args.data)
    if holdout < 1 or len(frame) - holdout < 104:
        raise ValueError("Keep at least two training years and a nonempty holdout")
    training = frame.iloc[:-holdout].copy()
    processor = Preprocessor.fit(training, int(config.get("fourier_order", 3)))
    sampling = Sampling(**config.get("sampling", {}))
    allocation = Allocation(**config.get("allocation", {}))
    args.output.mkdir(parents=True, exist_ok=False)
    processor.save(args.output / "preprocessor.json")
    trace = fit(training, processor, sampling)
    trace.to_netcdf(args.output / "posterior.nc")
    summary = az.summary(trace, var_names=["decay", "half_saturation", "slope", "media_beta", "control_beta", "intercept", "sigma"])
    summary.to_csv(args.output / "posterior_summary.csv")
    posterior = posterior_arrays(trace)
    means = predict_mean_draws(frame, processor, posterior)
    predictions = prediction_frame(frame, means, posterior["sigma"] * processor.revenue_scale, sampling.seed)
    predictions["split"] = ["training"] * len(training) + ["holdout"] * holdout
    predictions.to_csv(args.output / "predictions.csv", index=False)
    held = predictions.iloc[-holdout:]
    report = {"dataset_sha256": fingerprint(args.data), "rows": len(frame), "channels": 5,
              "training_rows": len(training), "holdout_weeks": holdout,
              "training_end": training.DATE.iloc[-1].date().isoformat(), "holdout_start": held.date.iloc[0].date().isoformat(),
              "metrics": metrics(held.actual, held.prediction, held.predictive_p05, held.predictive_p95),
              "diagnostics": {"divergences": int(trace.sample_stats.diverging.sum()),
                              "maximum_rhat": float(summary.r_hat.max()), "minimum_bulk_ess": float(summary.ess_bulk.min())},
              "sampling": asdict(sampling), "allocation_config": asdict(allocation),
              "prediction_context": "Holdout predictions condition on observed holdout spend and context; revenue is excluded from fitting"}
    write_json(args.output / "evaluation.json", report)
    subset = posterior_arrays(trace, allocation.posterior_draws, sampling.seed)
    write_json(args.output / "allocation.json", optimize_budget(training, processor, subset, allocation, sampling.seed))
    from .plots import plot_predictions
    plot_predictions(predictions, args.output / "holdout.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
