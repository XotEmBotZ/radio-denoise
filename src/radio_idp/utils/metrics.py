"""Metrics and evaluation utilities for radio-IDP."""

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from radio_idp.modulation import demap_symbols


@jax.jit
def compute_evm(tx: jax.Array, rx_hat: jax.Array) -> jax.Array:
    """Compute linear Error Vector Magnitude (EVM).

    Args:
        tx: Clean reference symbols of shape (..., L).
        rx_hat: Recovered/estimated symbols of shape (..., L).

    Returns:
        Scalar linear EVM: sqrt(E[|tx - rx_hat|^2] / E[|tx|^2]).
    """
    err_power = jnp.mean(jnp.abs(tx - rx_hat) ** 2)
    ref_power = jnp.mean(jnp.abs(tx) ** 2)
    return jnp.sqrt(err_power / jnp.maximum(ref_power, 1e-12))


@jax.jit
def compute_evm_percent(tx: jax.Array, rx_hat: jax.Array) -> jax.Array:
    """Compute EVM in percentage (%)."""
    return compute_evm(tx, rx_hat) * 100.0


@jax.jit
def compute_evm_db(tx: jax.Array, rx_hat: jax.Array) -> jax.Array:
    """Compute EVM in decibels (dB).

    Formula: 10 * log10(E[|tx - rx_hat|^2] / E[|tx|^2]).
    """
    err_power = jnp.mean(jnp.abs(tx - rx_hat) ** 2)
    ref_power = jnp.mean(jnp.abs(tx) ** 2)
    ratio = jnp.maximum(err_power / jnp.maximum(ref_power, 1e-12), 1e-12)
    return 10.0 * jnp.log10(ratio)


@jax.jit
def compute_ser(labels: jax.Array, pred_labels: jax.Array) -> jax.Array:
    """Compute Symbol Error Rate (SER).

    Args:
        labels: Ground truth integer symbol labels of shape (..., L).
        pred_labels: Predicted integer symbol labels of shape (..., L).

    Returns:
        Fraction of incorrectly predicted symbols in [0.0, 1.0].
    """
    return jnp.mean(labels != pred_labels)


@jax.jit
def demap_and_compute_ser(
    rx_hat: jax.Array,
    labels: jax.Array,
    constellation: jax.Array,
) -> jax.Array:
    """Demap recovered symbols and compute Symbol Error Rate (SER)."""
    pred_labels = demap_symbols(rx_hat, constellation)
    return compute_ser(labels, pred_labels)


@partial(jax.jit, static_argnames=("num_bits",))
def labels_to_bits(labels: jax.Array, num_bits: int) -> jax.Array:
    """Convert integer symbol indices to bit vectors.

    Args:
        labels: Array of integer labels (..., L).
        num_bits: Bits per constellation symbol (e.g. 2 for QPSK, 4 for 16QAM).

    Returns:
        Bit array of shape (..., L, num_bits) with values in {0, 1}.
    """
    bit_shifts = jnp.arange(num_bits - 1, -1, -1, dtype=jnp.int32)
    return (labels[..., None] >> bit_shifts) & 1


@partial(jax.jit, static_argnames=("num_bits",))
def compute_ber(
    labels: jax.Array,
    pred_labels: jax.Array,
    num_bits: int,
) -> jax.Array:
    """Compute Bit Error Rate (BER) between ground-truth and predicted labels.

    Args:
        labels: Ground truth integer symbol labels of shape (..., L).
        pred_labels: Predicted integer symbol labels of shape (..., L).
        num_bits: Bits per symbol (e.g. 2 for QPSK, 4 for 16QAM).

    Returns:
        Fraction of bit mismatches in [0.0, 1.0].
    """
    true_bits = labels_to_bits(labels, num_bits=num_bits)
    pred_bits = labels_to_bits(pred_labels, num_bits=num_bits)
    return jnp.mean(true_bits != pred_bits)


@partial(jax.jit, static_argnames=("num_bits",))
def demap_and_compute_ber(
    rx_hat: jax.Array,
    labels: jax.Array,
    constellation: jax.Array,
    num_bits: int,
) -> jax.Array:
    """Demap symbols and compute Bit Error Rate (BER)."""
    pred_labels = demap_symbols(rx_hat, constellation)
    return compute_ber(labels, pred_labels, num_bits=num_bits)


@jax.jit
def compute_param_mse(theta_true: jax.Array, theta_pred: jax.Array) -> jax.Array:
    """Compute mean squared error per parameter column of shape (P,)."""
    return jnp.mean((theta_true - theta_pred) ** 2, axis=0)


@jax.jit
def compute_param_rmse(theta_true: jax.Array, theta_pred: jax.Array) -> jax.Array:
    """Compute root mean squared error per parameter column of shape (P,)."""
    return jnp.sqrt(compute_param_mse(theta_true, theta_pred))


@jax.jit
def compute_param_nrmse(
    theta_true: jax.Array,
    theta_pred: jax.Array,
    lower_bounds: jax.Array,
    upper_bounds: jax.Array,
) -> jax.Array:
    """Compute Normalized Root Mean Squared Error (NRMSE) per parameter.

    Normalized by each parameter's operational range: RMSE / (upper - lower).
    """
    rmse = compute_param_rmse(theta_true, theta_pred)
    param_range = jnp.maximum(upper_bounds - lower_bounds, 1e-12)
    return rmse / param_range


def compute_param_metrics_dict(
    theta_true: np.ndarray | jax.Array,
    theta_pred: np.ndarray | jax.Array,
    param_names: list[str],
    lower_bounds: np.ndarray | jax.Array | None = None,
    upper_bounds: np.ndarray | jax.Array | None = None,
) -> dict[str, float]:
    """Compute per-parameter RMSE and optional NRMSE mapped to parameter names.

    Args:
        theta_true: Ground truth parameters (N, P).
        theta_pred: Predicted parameters (N, P).
        param_names: List of parameter string names.
        lower_bounds: Optional parameter lower bounds.
        upper_bounds: Optional parameter upper bounds.

    Returns:
        Dictionary mapping each param name to its RMSE (and NRMSE if bounds provided).
    """
    rmse = np.array(compute_param_rmse(jnp.array(theta_true), jnp.array(theta_pred)))
    metrics: dict[str, float] = {}

    for i, name in enumerate(param_names):
        metrics[f"rmse/{name}"] = float(rmse[i])

    metrics["rmse/mean"] = float(np.mean(rmse))

    if lower_bounds is not None and upper_bounds is not None:
        nrmse = np.array(
            compute_param_nrmse(
                jnp.array(theta_true),
                jnp.array(theta_pred),
                jnp.array(lower_bounds),
                jnp.array(upper_bounds),
            )
        )
        for i, name in enumerate(param_names):
            metrics[f"nrmse/{name}"] = float(nrmse[i])
        metrics["nrmse/mean"] = float(np.mean(nrmse))

    return metrics


def compute_binned_metrics(
    snr_values: np.ndarray,
    metric_values: np.ndarray,
    bin_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    """Bin metric values (such as BER, SER, or EVM) across SNR ranges.

    Args:
        snr_values: 1D array of burst SNR in dB (N,).
        metric_values: 1D array of burst-level metric values (N,).
        bin_edges: Array of SNR bin edges (e.g. [-5, 0, 5, 10, 15, 20, 25, 30]).

    Returns:
        Dictionary with:
            'snr_centers': Center of each SNR bin.
            'metric_mean': Mean metric value in each bin.
            'counts': Number of bursts in each bin.
    """
    bin_indices = np.digitize(snr_values, bin_edges) - 1
    num_bins = len(bin_edges) - 1

    snr_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    metric_mean = np.zeros(num_bins, dtype=np.float32)
    counts = np.zeros(num_bins, dtype=np.int32)

    for b in range(num_bins):
        mask = bin_indices == b
        counts[b] = np.sum(mask)
        if counts[b] > 0:
            metric_mean[b] = float(np.mean(metric_values[mask]))
        else:
            metric_mean[b] = np.nan

    return {
        "snr_centers": snr_centers,
        "metric_mean": metric_mean,
        "counts": counts,
    }
