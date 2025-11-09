def plot_predictions(frame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(11, 4))
    axis.plot(frame.date, frame.actual, color="#263c49", label="Observed revenue")
    axis.plot(frame.date, frame.prediction, color="#218578", label="Posterior mean")
    axis.fill_between(frame.date, frame.predictive_p05, frame.predictive_p95, alpha=.18, color="#218578", label="90% predictive interval")
    start = frame.loc[frame.split.eq("holdout"), "date"].iloc[0]
    axis.axvline(start, color="#c88041", linestyle="--", label="Holdout starts")
    axis.set(xlabel="Week", ylabel="Revenue", title="Chronological media mix validation")
    axis.legend(loc="best", frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)
