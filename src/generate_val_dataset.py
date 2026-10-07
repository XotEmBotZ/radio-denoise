"""Regenerate validation dataset with Cartesian DC offset parameters."""

import gc
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import xarray as xr
from scipy.stats import qmc
from tqdm import tqdm

import radio_idp

param_names: list[str] = [
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

lower_bounds: np.ndarray = np.array(
    [-5.0, -0.08, 0.001, -2.0, -10.0, 1.5, 0.75, -0.20, -0.20, 0.15, 0.0],
    dtype=np.float32,
)

upper_bounds: np.ndarray = np.array(
    [32.0, 0.08, 0.050, 2.0, 10.0, 4.0, 1.80, 0.20, 0.20, 0.80, 10.0],
    dtype=np.float32,
)


def generate_deterministic_burst(
    key: jax.Array,
    theta: jax.Array,
    constellation: jax.Array,
    sample_len: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    key_data, key_taps = jax.random.split(key, 2)
    labels = jax.random.randint(
        key_data, shape=(sample_len,), minval=0, maxval=constellation.shape[0]
    )
    tx = radio_idp.encode_symbols(labels, constellation)
    amp_imb = 10.0 ** (theta[3] / 20.0)
    phase_imb_rad = theta[4] * (jnp.pi / 180.0)
    dc_i = theta[7]
    dc_q = theta[8]
    taps = radio_idp.get_multipath_taps(
        key_taps, num_taps=3, decay=theta[9], k_factor=theta[10]
    )
    rx_det = radio_idp.deterministic_pipeline(
        tx,
        amp_imb=amp_imb,
        phase_imb_rad=phase_imb_rad,
        dc_i=dc_i,
        dc_q=dc_q,
        v_sat=theta[6],
        p=theta[5],
        taps=taps,
        cfo_norm=theta[1],
    )
    return tx, rx_det, labels


def apply_stochastic_noise(
    rx_det: jax.Array,
    key_noise: jax.Array,
    theta: jax.Array,
) -> jax.Array:
    return radio_idp.stochastic_pipeline(
        rx_det,
        rng=key_noise,
        phase_noise_std=theta[2],
        snr_db=theta[0],
    )


def generate_single_burst(
    key: jax.Array,
    theta: jax.Array,
    constellation: jax.Array,
    sample_len: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    key_det, key_noise = jax.random.split(key, 2)
    tx, rx_det, labels = generate_deterministic_burst(
        key_det, theta, constellation, sample_len
    )
    rx_full = apply_stochastic_noise(rx_det, key_noise, theta)
    return tx, rx_det, rx_full, labels


generate_burst = jax.vmap(generate_single_burst, in_axes=(0, 0, None, None))


def main() -> None:
    n_combinations = 2**16
    sampler = qmc.Sobol(d=len(param_names), scramble=True, rng=19956)
    unit_samples = sampler.random_base2(m=16)
    theta_matrix = np.array(
        qmc.scale(unit_samples, lower_bounds, upper_bounds), dtype=np.float32
    )

    master_key = jax.random.key(19956)
    key_train, key_val = jax.random.split(master_key, 2)
    burst_keys = jax.random.split(key_val, n_combinations)

    sample_length = 1024
    save_dir: Path = Path("data/validation")
    save_dir.mkdir(parents=True, exist_ok=True)

    constellations: dict[str, jax.Array] = {
        "qpsk": radio_idp.get_qpsk_constellation(),
        "8psk": radio_idp.get_psk_constellation(3),
        "16qam": radio_idp.get_qam_constellation(4),
        "64qam": radio_idp.get_qam_constellation(6),
        "16apsk": radio_idp.get_apsk_constellation(4),
        "32apsk": radio_idp.get_apsk_constellation(5),
    }

    for mod_name, const in tqdm(constellations.items(), desc="Generating validation sets"):
        tx, rx_det, rx_full, labels = generate_burst(
            burst_keys, theta_matrix, const, sample_length
        )
        ds = xr.Dataset(
            data_vars={
                "tx": (["burst", "symbol"], np.array(tx)),
                "rx_det": (["burst", "symbol"], np.array(rx_det)),
                "rx_full": (["burst", "symbol"], np.array(rx_full)),
                "labels": (["burst", "symbol"], np.array(labels)),
                "params": (["burst", "param"], np.array(theta_matrix)),
                "constellation": (["point"], np.array(const)),
            },
            coords={
                "burst": np.arange(n_combinations),
                "symbol": np.arange(sample_length),
                "param": param_names,
                "point": np.arange(const.shape[0]),
            },
            attrs={
                "modulation": mod_name,
                "num_symbols": sample_length,
                "num_bursts": n_combinations,
                "seed": 19956,
            },
        )
        ds.to_netcdf(save_dir / f"val_sobol_{mod_name}_32k.nc", engine="h5netcdf")
        del tx, rx_det, rx_full, labels, ds
        gc.collect()


if __name__ == "__main__":
    main()
