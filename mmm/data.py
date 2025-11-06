from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

CHANNELS = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"]
CONTROLS = ["competitor_sales_B", "newsletter"]
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "datasets" / "robyn_weekly.csv"


def load_data(path=DEFAULT_DATA):
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"DATE", "revenue", "events", *CHANNELS, *CONTROLS}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {required - set(frame.columns)}")
    frame["DATE"] = pd.to_datetime(frame["DATE"], errors="raise")
    frame = frame.sort_values("DATE").reset_index(drop=True)
    if frame["DATE"].duplicated().any() or not frame["DATE"].diff().dropna().eq(pd.Timedelta(days=7)).all():
        raise ValueError("Expected unique, contiguous weekly dates")
    values = frame[["revenue", *CHANNELS, *CONTROLS]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (frame[CHANNELS].to_numpy() < 0).any() or (frame.revenue <= 0).any():
        raise ValueError("Finite values, nonnegative spend, and positive revenue required")
    frame["events"] = frame["events"].astype(str)
    return frame


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Preprocessor:
    spend_scale: list
    revenue_scale: float
    control_mean: list
    control_scale: list
    event_levels: list
    origin: str
    trend_center: float
    trend_scale: float
    fourier_order: int = 3

    @classmethod
    def fit(cls, training, fourier_order=3):
        if fourier_order < 1 or fourier_order > 10:
            raise ValueError("Fourier order must be between 1 and 10")
        spends = training[CHANNELS].to_numpy(float)
        scale = np.quantile(spends, .9, axis=0)
        if np.any(scale <= 0):
            raise ValueError("Each channel needs positive training spend at its 90th percentile")
        controls = training[CONTROLS].to_numpy(float)
        std = controls.std(axis=0)
        origin = training.DATE.iloc[0]
        time = (training.DATE - origin).dt.days.to_numpy() / 365.25
        return cls(scale.tolist(), float(training.revenue.mean()), controls.mean(axis=0).tolist(),
                   np.where(std > np.maximum(1, np.abs(controls.mean(axis=0))) * 1e-10, std, 1).tolist(), sorted(training.events.unique().tolist()),
                   origin.isoformat(), float(time.mean()), float(time.std() or 1), fourier_order)

    @property
    def feature_names(self):
        return (["trend"] + [f"annual_{trig}_{k}" for k in range(1, self.fourier_order + 1) for trig in ("sin", "cos")]
                + CONTROLS + [f"event_{level}" for level in self.event_levels[1:]])

    def transform(self, frame):
        time = (frame.DATE - pd.Timestamp(self.origin)).dt.days.to_numpy(float) / 365.25
        terms = [(time - self.trend_center) / self.trend_scale]
        for k in range(1, self.fourier_order + 1):
            terms.extend([np.sin(2 * np.pi * k * time), np.cos(2 * np.pi * k * time)])
        controls = (frame[CONTROLS].to_numpy(float) - self.control_mean) / self.control_scale
        terms.extend(controls[:, i] for i in range(len(CONTROLS)))
        terms.extend(frame.events.eq(level).to_numpy(float) for level in self.event_levels[1:])
        return frame[CHANNELS].to_numpy(float) / self.spend_scale, np.column_stack(terms)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
