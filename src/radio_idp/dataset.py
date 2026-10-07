"""Dataset and on-the-fly streaming data generator for radio-IDP.

This module provides:
1. Physical impairment bounds and Randomized Sobol (QMC) parameter sampler.
2. JIT-compiled on-the-fly streaming synthetic burst generator for training.
3. Domain randomization parameter perturbation for Stage 2 robustness.
4. NetCDF validation dataset reader for benchmark evaluation.
"""

from pathlib import Path
from typing import Iterator, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc
import xarray as xr

import radio_idp

# --- Physical Impairment Parameter Definitions ---
PARAM_NAMES: list[str] = [
    "snr_db",
    "cfo_norm",
    "phase_noise_std",
    "iq_amp_imbalance_db",
    "iq_phase_imbalance_deg",
    "rapp_p",
    "rapp_v_sat",
    "dc_i",
    "dc_q",
    "multipath_decay",
    "multipath_k_factor",
]

LOWER_BOUNDS: jnp.ndarray = jnp.array(
    [-5.0, -0.08, 0.001, -2.0, -10.0, 1.5, 0.75, -0.20, -0.20, 0.15, 0.0],
    dtype=jnp.float32,
)

UPPER_BOUNDS: jnp.ndarray = jnp.array(
    [32.0, 0.08, 0.050, 2.0, 10.0, 4.0, 1.80, 0.20, 0.20, 0.80, 10.0],
    dtype=jnp.float32,
)


class TrainingBatch(NamedTuple):
    """Container for a generated training batch (registered JAX PyTree)."""

    tx: jax.Array  # Clean transmitted complex symbols (B, L)
    rx_full: jax.Array  # Fully impaired received symbols (B, L)
    rx_cleaned: jax.Array  # Inverted using jittered/estimated theta (B, L) for Stage 2
    theta: jax.Array  # True physical parameters (B, 11) for Stage 1
    theta_jittered: jax.Array  # Perturbed physical parameters (B, 11)
    labels: jax.Array  # Ground truth symbol integer labels (B, L)


# --- Step 1: Parameter Sampling via Randomized Sobol (QMC) ---


def init_sobol_table(
    m_power: int = 18,
    seed: int = 19956,
) -> jax.Array:
    """Pre-compute a scrambled Sobol low-discrepancy table in unit cube [0, 1)^11.

    Args:
        n_points: Total points to generate (default 262,144 points, ~11.5 MB VRAM).
        seed: Scramble seed for reproducibility.

    Returns:
        Device array of shape (n_points, 11) with dtype float32.
    """
    sampler = qmc.Sobol(d=len(LOWER_BOUNDS), scramble=True, rng=seed)
    samples = sampler.random_base2(m_power)
    return jnp.array(samples, dtype=jnp.float32)


def sample_parameters_sobol(
    key: jax.Array,
    sobol_table: jax.Array,
    indices: jax.Array,
) -> jax.Array:
    """Sample continuous parameters via Cranley-Patterson shifted Sobol sequence.

    Args:
        key: JAX PRNG key for generating random shift delta.
        sobol_table: Pre-computed base Sobol table of shape (N, 11).
        indices: Array of integer indices of shape (batch_size,) to slice from table.

    Returns:
        Array of shape (batch_size, 11) with physical impairment values.
    """
    u_base = sobol_table[indices]
    delta = jax.random.uniform(key, shape=(1, 11), dtype=jnp.float32)
    u_shifted = (u_base + delta) % 1  # to confind points to [0,1]
    return LOWER_BOUNDS + u_shifted * (UPPER_BOUNDS - LOWER_BOUNDS)


def perturb_parameters(
    key: jax.Array,
    theta: jax.Array,
    sigma: float = 0.05,
) -> jax.Array:
    """Apply Gaussian jitter in normalized parameter space for domain randomization.

    Args:
        key: JAX PRNG key.
        theta: Ground truth parameters of shape (B, 11).
        sigma: Relative standard deviation of perturbation in normalized [0, 1] space.

    Returns:
        Perturbed theta of shape (B, 11), clamped within valid physical bounds.
    """
    theta_norm = (theta - LOWER_BOUNDS) / (UPPER_BOUNDS - LOWER_BOUNDS)
    noise = jax.random.normal(key, shape=theta.shape, dtype=jnp.float32) * sigma
    theta_new = jnp.clip(theta_norm + noise, 0, 1)
    return LOWER_BOUNDS + theta_new * (UPPER_BOUNDS - LOWER_BOUNDS)


# --- Step 2: Burst Synthesis & Deterministic Denoising ---


def unpack_theta(
    theta: jax.Array,
    key_taps: jax.Array,
) -> dict[str, jax.Array]:
    """Convert raw 11-element theta into named physical arguments."""
    amp_imb = 10.0 ** (theta[3] / 20.0)
    phase_imb_rad = theta[4] * (jnp.pi / 180.0)
    dc_i = theta[7]
    dc_q = theta[8]
    taps = radio_idp.get_multipath_taps(
        key_taps, num_taps=3, decay=theta[9], k_factor=theta[10]
    )
    return {
        "snr_db": theta[0],
        "cfo_norm": theta[1],
        "phase_noise_std": theta[2],
        "amp_imb": amp_imb,
        "phase_imb_rad": phase_imb_rad,
        "rapp_p": theta[5],
        "rapp_v_sat": theta[6],
        "dc_i": dc_i,
        "dc_q": dc_q,
        "taps": taps,
    }


def generate_single_burst(
    key: jax.Array,
    theta: jax.Array,
    constellation: jax.Array,
    sample_len: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Synthesize a single transmitted and impaired burst.

    Args:
        key: JAX PRNG key.
        theta: Parameter vector of shape (11,).
        constellation: Array of complex constellation points.
        sample_len: Number of symbols in the burst (e.g. 1024).

    Returns:
        tuple of (tx, rx_det, rx_full, labels, taps)
    """
    key_data, key_taps, key_noise = jax.random.split(key, 3)
    labels = jax.random.randint(
        key_data, shape=(sample_len,), minval=0, maxval=constellation.shape[0], dtype=jnp.int32
    )
    tx = constellation[labels]
    params = unpack_theta(theta, key_taps)
    rx_det = radio_idp.deterministic_pipeline(
        tx,
        amp_imb=params["amp_imb"],
        phase_imb_rad=params["phase_imb_rad"],
        dc_i=params["dc_i"],
        dc_q=params["dc_q"],
        v_sat=params["rapp_v_sat"],
        p=params["rapp_p"],
        taps=params["taps"],
        cfo_norm=params["cfo_norm"],
    )
    rx_full = radio_idp.stochastic_pipeline(
        rx_det,
        rng=key_noise,
        phase_noise_std=params["phase_noise_std"],
        snr_db=params["snr_db"],
    )
    return tx, rx_det, rx_full, labels, params["taps"]


def invert_burst(
    rx_full: jax.Array,
    theta: jax.Array,
    taps: jax.Array,
) -> jax.Array:
    """Invert deterministic impairments from rx_full using given theta.

    Args:
        rx_full: Received burst symbols of shape (L,).
        theta: Physical parameters of shape (11,).
        taps: Multipath channel taps.

    Returns:
        Cleaned symbols of shape (L,) ready for Stage 2 flow denoiser.
    """
    amp_imb = 10.0 ** (theta[3] / 20.0)
    phase_imb_rad = theta[4] * (jnp.pi / 180.0)
    dc_i = theta[7]
    dc_q = theta[8]
    return radio_idp.deterministic_denoise_pipeline(
        rx_full,
        cfo_norm=theta[1],
        taps=taps,
        v_sat=theta[6],
        p=theta[5],
        dc_i=dc_i,
        dc_q=dc_q,
        amp_imb=amp_imb,
        phase_imb_rad=phase_imb_rad,
    )


# --- Step 3: Vectorized Training Batch Generator ---


def generate_training_batch(
    key: jax.Array,
    sobol_table: jax.Array,
    indices: jax.Array,
    constellation: jax.Array,
    sample_len: int = 1024,
    jitter_sigma: float = 0.05,
) -> TrainingBatch:
    """Vectorized generator producing a full TrainingBatch on device."""
    key_sobol, key_jitter, key_bursts = jax.random.split(key, 3)
    batch_size = indices.shape[0]
    theta = sample_parameters_sobol(key_sobol, sobol_table, indices)
    theta_jittered = perturb_parameters(key_jitter, theta, sigma=jitter_sigma)
    batch_keys = jax.random.split(key_bursts, batch_size)
    tx, rx_det, rx_full, labels, taps = jax.vmap(
        generate_single_burst, in_axes=(0, 0, None, None)
    )(batch_keys, theta, constellation, sample_len)
    rx_cleaned = jax.vmap(invert_burst, in_axes=(0, 0, 0))(rx_full, theta_jittered, taps)

    return TrainingBatch(
        tx=tx,
        rx_full=rx_full,
        rx_cleaned=rx_cleaned,
        theta=theta,
        theta_jittered=theta_jittered,
        labels=labels,
    )


# --- Step 4: Streaming Iterator Interface ---


class StreamingBurstGenerator:
    """Stateful iterator wrapping generate_training_batch for training loops."""

    def __init__(
        self,
        seed: int,
        constellation: jax.Array,
        batch_size: int = 64,
        sample_len: int = 1024,
        jitter_sigma: float = 0.05,
        sobol_points: int = 2**18,
    ):
        self.key = jax.random.key(seed)
        self.constellation = constellation
        self.batch_size = batch_size
        self.sample_len = sample_len
        self.jitter_sigma = jitter_sigma
        self.sobol_points = sobol_points
        m_power = int(round(np.log2(sobol_points)))
        self.sobol_table = init_sobol_table(m_power=m_power, seed=seed)
        self._generate_fn = jax.jit(
            generate_training_batch, static_argnames=("sample_len",)
        )

    def __iter__(self) -> Iterator[TrainingBatch]:
        return self

    def __next__(self) -> TrainingBatch:
        self.key, key_batch, key_idx = jax.random.split(self.key, 3)
        indices = jax.random.randint(
            key_idx, shape=(self.batch_size,), minval=0, maxval=self.sobol_points
        )
        return self._generate_fn(
            key_batch,
            self.sobol_table,
            indices,
            self.constellation,
            sample_len=self.sample_len,
            jitter_sigma=self.jitter_sigma,
        )


# --- Step 5: Validation NetCDF Reader ---


def load_validation_bursts(
    file_path: Path | str,
    max_bursts: int | None = None,
) -> dict[str, np.ndarray]:
    """Load a slice of pre-generated validation data from NetCDF.

    Args:
        file_path: Path to val_sobol_*.nc file.
        max_bursts: Optional limit on number of bursts to read into memory.

    Returns:
        Dictionary with arrays: tx, rx_det, rx_full, labels, params, constellation.
    """
    with xr.open_dataset(file_path, engine="h5netcdf") as ds:
        if max_bursts is not None:
            ds = ds.isel(burst=slice(0, max_bursts))
        return {
            "tx": np.array(ds["tx"].values),
            "rx_det": np.array(ds["rx_det"].values),
            "rx_full": np.array(ds["rx_full"].values),
            "labels": np.array(ds["labels"].values),
            "params": np.array(ds["params"].values),
            "constellation": np.array(ds["constellation"].values),
        }
