def is_anomaly(mse_err: float, threshold: float) -> bool:
    """Return True if the reconstruction MSE exceeds the given threshold.

    Args:
        mse_err: Mean‑squared error between input and reconstruction.
        threshold: Configurable threshold for anomaly detection.
    """
    return mse_err > threshold
