"""Tier-2 Benchmarking Suite: Publication-grade BER/SER and EVM waterfall evaluation."""

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Callable

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import radio_idp
from radio_idp.dataset import LOWER_BOUNDS, PARAM_NAMES, UPPER_BOUNDS, invert_burst, load_validation_bursts
from radio_idp.utils.checkpoint import ModelCheckpointManager
from radio_idp.utils.metrics import (
    compute_ber,
    compute_binned_metrics,
    compute_evm_db,
    compute_param_metrics_dict,
    compute_ser,
    demap_symbols,
)


@dataclass
class BenchmarkConfig:
    """Configuration for Tier-2 benchmark evaluation."""

    val_dir: str = "data/validation"
    modulations: list[str] = field(
        default_factory=lambda: ["qpsk", "8psk", "16qam", "64qam", "16apsk", "32apsk"]
    )
    max_bursts_per_mod: int | None = 8192
    batch_size: int = 128
    ode_steps: int = 5
    ode_solver: str = "euler"
    snr_bins: list[float] = field(
        default_factory=lambda: [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
    )
    output_dir: str = "benchmark_results"


def complex_to_real_channels(symbols: jax.Array) -> jax.Array:
    """Stack real and imaginary components into feature dimension."""
    return jnp.stack([symbols.real, symbols.imag], axis=-1)


def real_channels_to_complex(channels: jax.Array) -> jax.Array:
    """Convert real and imaginary feature channels back to complex symbols."""
    return channels[..., 0] + 1j * channels[..., 1]


def solve_ode(
    model: nnx.Module,
    cond: jax.Array,
    num_steps: int = 5,
    solver: str = "euler",
) -> jax.Array:
    """Integrate velocity field from t=0 to t=1 using numerical ODE integration."""
    batch_size = cond.shape[0]
    dt = 1.0 / num_steps
    x = cond

    for step in range(num_steps):
        t_val = step * dt
        t_arr = jnp.full((batch_size, 1, 1), t_val, dtype=jnp.float32)
        x_cat = jnp.concatenate([x, cond], axis=-1)
        v = model(x_cat, t_arr)

        if solver == "euler":
            x = x + dt * v
        elif solver == "heun":
            x_pred = x + dt * v
            t_next = jnp.full((batch_size, 1, 1), t_val + dt, dtype=jnp.float32)
            x_cat_next = jnp.concatenate([x_pred, cond], axis=-1)
            v_next = model(x_cat_next, t_next)
            x = x + 0.5 * dt * (v + v_next)
        else:
            x = x + dt * v

    return x


def bits_per_symbol(modulation: str) -> int:
    """Return number of bits per symbol for a given modulation order."""
    mapping = {
        "qpsk": 2,
        "8psk": 3,
        "16qam": 4,
        "64qam": 6,
        "16apsk": 4,
        "32apsk": 5,
    }
    return mapping.get(modulation.lower(), 2)


def run_benchmark(
    stage1_model: nnx.Module | None = None,
    stage2_model: nnx.Module | None = None,
    config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Run comprehensive Tier-2 benchmark across validation datasets.

    Args:
        stage1_model: Optional trained parameter estimator.
        stage2_model: Optional trained Flow Matching denoiser.
        config: Optional benchmark configuration.

    Returns:
        Structured dictionary of evaluation results and curves.
    """
    if config is None:
        config = BenchmarkConfig()

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    bin_edges = np.array(config.snr_bins, dtype=np.float32)
    taps_default = jnp.array([1.0], dtype=jnp.complex64)

    results: dict[str, Any] = {
        "config": {
            "ode_steps": config.ode_steps,
            "ode_solver": config.ode_solver,
            "snr_bins": config.snr_bins,
        },
        "modulations": {},
    }

    val_dir = Path(config.val_dir)

    for mod in config.modulations:
        nc_file = val_dir / f"val_sobol_{mod}_32k.nc"
        if not nc_file.exists():
            continue

        val_data = load_validation_bursts(nc_file, max_bursts=config.max_bursts_per_mod)
        tx_all = val_data["tx"]
        rx_full_all = val_data["rx_full"]
        labels_all = val_data["labels"]
        params_all = val_data["params"]
        constellation = jnp.array(val_data["constellation"])
        num_bits = bits_per_symbol(mod)
        num_bursts = len(tx_all)

        burst_evm_list = []
        burst_ser_list = []
        burst_ber_list = []
        snr_list = []
        latencies_us = []

        for i in range(0, num_bursts, config.batch_size):
            batch_tx = jnp.array(tx_all[i : i + config.batch_size])
            batch_rx = jnp.array(rx_full_all[i : i + config.batch_size])
            batch_labels = jnp.array(labels_all[i : i + config.batch_size])
            batch_params = jnp.array(params_all[i : i + config.batch_size])
            batch_snr = np.array(batch_params[:, 0])

            start_t = time.perf_counter()

            # 1. Deterministic inversion: either via Stage 1 model or ground-truth params
            if stage1_model is not None:
                x_in = jax.vmap(radio_idp.extract_features)(batch_rx)
                pred_u = stage1_model(x_in)
                theta_est = LOWER_BOUNDS + pred_u * (UPPER_BOUNDS - LOWER_BOUNDS)
                rx_cleaned = jax.vmap(invert_burst, in_axes=(0, 0, None))(
                    batch_rx, theta_est, taps_default
                )
            else:
                rx_cleaned = jax.vmap(invert_burst, in_axes=(0, 0, None))(
                    batch_rx, batch_params, taps_default
                )

            # 2. Stage 2 flow denoising if model provided, otherwise raw cleaned
            if stage2_model is not None:
                cond = complex_to_real_channels(rx_cleaned)
                denoised_channels = solve_ode(
                    stage2_model,
                    cond,
                    num_steps=config.ode_steps,
                    solver=config.ode_solver,
                )
                rx_recovered = real_channels_to_complex(denoised_channels)
            else:
                rx_recovered = rx_cleaned

            end_t = time.perf_counter()
            batch_latency_us = ((end_t - start_t) / len(batch_tx)) * 1e6
            latencies_us.append(batch_latency_us)

            # Compute burst-level metrics
            for b in range(len(batch_tx)):
                tx_b = batch_tx[b]
                rx_b = rx_recovered[b]
                lbl_b = batch_labels[b]

                evm_db = float(compute_evm_db(tx_b, rx_b))
                pred_lbl_b = demap_symbols(rx_b, constellation)
                ser = float(compute_ser(lbl_b, pred_lbl_b))
                ber = float(compute_ber(lbl_b, pred_lbl_b, num_bits=num_bits))

                burst_evm_list.append(evm_db)
                burst_ser_list.append(ser)
                burst_ber_list.append(ber)
                snr_list.append(float(batch_snr[b]))

        burst_evm_arr = np.array(burst_evm_list)
        burst_ser_arr = np.array(burst_ser_list)
        burst_ber_arr = np.array(burst_ber_list)
        snr_arr = np.array(snr_list)

        binned_evm = compute_binned_metrics(snr_arr, burst_evm_arr, bin_edges)
        binned_ser = compute_binned_metrics(snr_arr, burst_ser_arr, bin_edges)
        binned_ber = compute_binned_metrics(snr_arr, burst_ber_arr, bin_edges)

        results["modulations"][mod] = {
            "snr_centers": binned_ber["snr_centers"].tolist(),
            "ber_mean": binned_ber["metric_mean"].tolist(),
            "ser_mean": binned_ser["metric_mean"].tolist(),
            "evm_db_mean": binned_evm["metric_mean"].tolist(),
            "mean_latency_us": float(np.mean(latencies_us)),
            "overall_ber": float(np.mean(burst_ber_arr)),
            "overall_evm_db": float(np.mean(burst_evm_arr)),
        }

    # Save summary results
    results_file = output_path / "benchmark_summary.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_benchmark_curves(results, output_path)
    return results


def plot_benchmark_curves(results: dict[str, Any], output_path: Path) -> None:
    """Generate publication-ready waterfall BER vs SNR and EVM vs SNR plots."""
    modulations = results.get("modulations", {})
    if not modulations:
        return

    # 1. BER vs SNR Waterfall
    plt.figure(figsize=(8, 6))
    for mod, data in modulations.items():
        snrs = data["snr_centers"]
        bers = data["ber_mean"]
        # Filter out NaN or zero values for log plot
        valid = [(s, b) for s, b in zip(snrs, bers) if not np.isnan(b) and b > 0]
        if valid:
            x_vals, y_vals = zip(*valid)
            plt.semilogy(x_vals, y_vals, marker="o", label=mod.upper())

    plt.title("Bit Error Rate (BER) Waterfall vs. SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "waterfall_ber_vs_snr.png", dpi=300)
    plt.close()

    # 2. EVM vs SNR
    plt.figure(figsize=(8, 6))
    for mod, data in modulations.items():
        snrs = data["snr_centers"]
        evms = data["evm_db_mean"]
        valid = [(s, e) for s, e in zip(snrs, evms) if not np.isnan(e)]
        if valid:
            x_vals, y_vals = zip(*valid)
            plt.plot(x_vals, y_vals, marker="s", label=mod.upper())

    plt.title("Error Vector Magnitude (EVM) vs. SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("EVM (dB)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "evm_vs_snr.png", dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier-2 Waterfall Benchmark Evaluation")
    parser.add_argument("--val-dir", type=str, default="data/validation")
    parser.add_argument("--output-dir", type=str, default="benchmark_results")
    parser.add_argument("--ode-steps", type=int, default=5)
    parser.add_argument("--ode-solver", type=str, default="euler", choices=["euler", "heun"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-bursts", type=int, default=4096)
    args = parser.parse_args()

    config = BenchmarkConfig(
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        ode_steps=args.ode_steps,
        ode_solver=args.ode_solver,
        batch_size=args.batch_size,
        max_bursts_per_mod=args.max_bursts,
    )

    print(f"Running benchmark with baseline physics inversion on {config.val_dir}...")
    run_benchmark(config=config)
    print(f"Benchmark finished! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
