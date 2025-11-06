import numpy as np


def adstock(spend, decay, initial=None):
    spend, decay = np.asarray(spend, dtype=float), np.asarray(decay, dtype=float)
    if spend.ndim != 2 or decay.shape != (spend.shape[1],) or np.any((decay < 0) | (decay >= 1)) or np.any(spend < 0):
        raise ValueError("Expected nonnegative time-by-channel spend and decay in [0,1)")
    previous = np.zeros(spend.shape[1]) if initial is None else np.asarray(initial, dtype=float).copy()
    if previous.shape != decay.shape:
        raise ValueError("Initial carryover shape mismatch")
    result = np.empty_like(spend)
    for t in range(len(spend)):
        previous = spend[t] + decay * previous
        result[t] = previous
    return result


def hill(values, half_saturation, slope):
    values = np.asarray(values, dtype=float)
    half_saturation, slope = np.asarray(half_saturation), np.asarray(slope)
    if np.any(values < 0) or np.any(half_saturation <= 0) or np.any(slope <= 0):
        raise ValueError("Invalid saturation parameters")
    powered = np.power(values, slope)
    return powered / (powered + np.power(half_saturation, slope))


def symbolic_adstock(spend, decay):
    import pytensor
    import pytensor.tensor as pt
    values = pt.as_tensor_variable(np.asarray(spend, dtype="float64"))
    result, _ = pytensor.scan(
        fn=lambda current, previous, alpha: current + alpha * previous,
        sequences=[values], outputs_info=[pt.zeros(spend.shape[1], dtype="float64")],
        non_sequences=[decay], strict=True,
    )
    return result
