"""Stage 2 training engine: Stochastic Flow Matching Denoiser."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

import radio_idp
from radio_idp.dataset import (
    StreamingBurstGenerator,
    invert_burst,
    load_validation_bursts,
)
from radio_idp.utils.checkpoint import ModelCheckpointManager
from radio_idp.utils.logger import ExperimentLogger
from radio_idp.utils.metrics import compute_evm_db


@dataclass
class Stage2TrainConfig:
    """Hyperparameters and configuration for Stage 2 Flow Matching training."""

    num_steps: int = 10_000
    batch_size: int = 64
    sample_len: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    val_interval: int = 500
    val_bursts: int = 1024
    val_ode_steps: int = 5
    val_file: str = "data/validation/val_sobol_qpsk_32k.nc"
    checkpoint_dir: str = "checkpoints/stage2"
    log_dir: str = "runs"
    experiment_name: str = "stage2_flow_denoiser"
    seed: int = 42
    jitter_sigma: float = 0.05
    sobol_points: int = 2**18


def complex_to_real_channels(symbols: jax.Array) -> jax.Array:
    """Convert complex array (..., L) to real channels (..., L, 2)."""
    return jnp.stack([symbols.real, symbols.imag], axis=-1)


def real_channels_to_complex(channels: jax.Array) -> jax.Array:
    """Convert real channels (..., L, 2) back to complex array (..., L)."""
    return channels[..., 0] + 1j * channels[..., 1]


def stage2_loss_fn(
    model: nnx.Module,
    x_0: jax.Array,
    x_1: jax.Array,
    cond: jax.Array,
    key_t: jax.Array,
) -> jax.Array:
    """Compute Flow Matching velocity matching objective: E[||v_t - (x_1 - x_0)||^2]."""
    batch_size = x_0.shape[0]
    t = jax.random.uniform(key_t, shape=(batch_size, 1, 1), dtype=jnp.float32)

    # Optimal transport linear interpolation
    x_t = (1.0 - t) * x_0 + t * x_1
    target_velocity = x_1 - x_0

    # Stack flow state x_t and condition signal along feature channel
    x_cat = jnp.concatenate([x_t, cond], axis=-1)
    predicted_velocity = model(x_cat, t)

    return jnp.mean((predicted_velocity - target_velocity) ** 2)


@nnx.jit
def stage2_train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    x_0: jax.Array,
    x_1: jax.Array,
    cond: jax.Array,
    key_t: jax.Array,
) -> jax.Array:
    """Single JIT-compiled optimization step for Stage 2."""
    grad_fn = nnx.value_and_grad(stage2_loss_fn)
    loss, grads = grad_fn(model, x_0, x_1, cond, key_t)
    optimizer.update(model, grads)
    return loss


def solve_ode_euler(
    model: nnx.Module,
    cond: jax.Array,
    num_steps: int = 5,
) -> jax.Array:
    """Integrate velocity field from t=0 to t=1 using Euler integration.

    Args:
        model: Flow matching velocity network.
        cond: Conditioning signal of shape (B, L, 2) (e.g. rx_cleaned).
        num_steps: Number of integration steps (NFE).

    Returns:
        Denoised signal channels of shape (B, L, 2).
    """
    batch_size = cond.shape[0]
    dt = 1.0 / num_steps
    x = cond

    for step in range(num_steps):
        t_val = step * dt
        t_arr = jnp.full((batch_size, 1, 1), t_val, dtype=jnp.float32)
        x_cat = jnp.concatenate([x, cond], axis=-1)
        v = model(x_cat, t_arr)
        x = x + dt * v

    return x


def evaluate_stage2(
    model: nnx.Module,
    val_data: dict[str, np.ndarray],
    num_ode_steps: int = 5,
    batch_size: int = 128,
) -> dict[str, float]:
    """Evaluate Stage 2 Flow Denoiser on validation set and compute EVM in dB."""
    tx_all = val_data["tx"]
    rx_full_all = val_data["rx_full"]
    params_all = val_data["params"]
    num_samples = len(tx_all)

    taps_default = jnp.array([1.0], dtype=jnp.complex64)
    all_evm_db = []

    for i in range(0, num_samples, batch_size):
        batch_tx = jnp.array(tx_all[i : i + batch_size])
        batch_rx = jnp.array(rx_full_all[i : i + batch_size])
        batch_params = jnp.array(params_all[i : i + batch_size])

        # Apply deterministic inversion with true theta as baseline
        rx_cleaned = jax.vmap(invert_burst, in_axes=(0, 0, None))(
            batch_rx, batch_params, taps_default
        )

        cond = complex_to_real_channels(rx_cleaned)
        denoised_channels = solve_ode_euler(model, cond, num_steps=num_ode_steps)
        rx_recovered = real_channels_to_complex(denoised_channels)

        batch_evm = float(compute_evm_db(batch_tx, rx_recovered))
        all_evm_db.append(batch_evm)

    mean_evm_db = float(np.mean(all_evm_db))
    return {
        "evm_db": mean_evm_db,
    }


def train_stage2(
    model: nnx.Module,
    config: Stage2TrainConfig | None = None,
) -> dict[str, Any]:
    """Train a Stage 2 Flow Matching Denoiser model with Flax NNX.

    Args:
        model: Any nnx.Module accepting input of shape (B, L, 4) and t (B, 1, 1),
               outputting velocity (B, L, 2).
        config: Optional configuration dataclass.

    Returns:
        Dictionary containing trained model, best validation EVM (dB), and run directory.
    """
    if config is None:
        config = Stage2TrainConfig()

    logger = ExperimentLogger(
        log_dir=config.log_dir,
        experiment_name=config.experiment_name,
    )
    checkpoint_mgr = ModelCheckpointManager(
        directory=config.checkpoint_dir,
        max_to_keep=3,
    )

    warmup = min(config.warmup_steps, max(1, config.num_steps // 10))
    decay_steps = max(config.num_steps, warmup + 1)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=config.learning_rate,
        warmup_steps=warmup,
        decay_steps=decay_steps,
        end_value=1e-6,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    constellation = radio_idp.get_qpsk_constellation()
    generator = StreamingBurstGenerator(
        seed=config.seed,
        constellation=constellation,
        batch_size=config.batch_size,
        sample_len=config.sample_len,
        jitter_sigma=config.jitter_sigma,
        sobol_points=config.sobol_points,
    )

    val_data = load_validation_bursts(config.val_file, max_bursts=config.val_bursts)

    best_evm_db = float("inf")
    key_rng = jax.random.key(config.seed)
    pbar = tqdm(range(1, config.num_steps + 1), desc="Stage 2 Training")

    for step in pbar:
        batch = next(generator)
        key_rng, key_step = jax.random.split(key_rng)

        x_1 = complex_to_real_channels(batch.tx)
        x_0 = complex_to_real_channels(batch.rx_cleaned)
        cond = x_0

        loss = stage2_train_step(model, optimizer, x_0, x_1, cond, key_step)
        loss_val = float(loss)

        if step % 50 == 0:
            logger.log_metrics(step, {"flow_loss": loss_val}, prefix="train")
            pbar.set_postfix({"flow_loss": f"{loss_val:.5f}"})

        if step % config.val_interval == 0 or step == config.num_steps:
            val_metrics = evaluate_stage2(
                model, val_data, num_ode_steps=config.val_ode_steps
            )
            logger.log_metrics(step, val_metrics, prefix="val")
            val_evm = val_metrics.get("evm_db", float("inf"))

            if val_evm < best_evm_db:
                best_evm_db = val_evm
                checkpoint_mgr.save(
                    step,
                    model,
                    opt_state=optimizer,
                    extra_metadata={"val_evm_db": best_evm_db},
                )

            pbar.set_postfix({
                "flow_loss": f"{loss_val:.5f}",
                "val_evm_db": f"{val_evm:.2f}dB",
            })

    checkpoint_mgr.wait_until_finished()
    logger.close()

    return {
        "model": model,
        "best_evm_db": best_evm_db,
        "run_dir": str(logger.run_dir),
    }
