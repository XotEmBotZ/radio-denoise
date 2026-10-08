"""Stage 1 training engine: Deterministic Parameter Estimator."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

import radio_idp
from radio_idp.dataset import (
    LOWER_BOUNDS,
    PARAM_NAMES,
    UPPER_BOUNDS,
    generate_single_burst,
    init_sobol_table,
    load_validation_bursts,
    sample_parameters_sobol,
)
from radio_idp.utils.checkpoint import ModelCheckpointManager
from radio_idp.utils.logger import ExperimentLogger
from radio_idp.utils.metrics import compute_param_metrics_dict, compute_param_nrmse


@dataclass
class Stage1TrainConfig:
    """Hyperparameters and execution configuration for Stage 1 training."""

    num_steps: int = 10_000
    batch_size: int = 64
    sample_len: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    val_interval: int = 500
    val_bursts: int = 2048
    val_file: str = "data/validation/val_sobol_qpsk_32k.nc"
    checkpoint_dir: str = "checkpoints/stage1"
    log_dir: str = "runs"
    experiment_name: str = "stage1_estimator"
    seed: int = 42
    sobol_points: int = 2**18



def stage1_loss_fn(
    model: nnx.Module,
    x_in: jax.Array,
    theta_norm_target: jax.Array,
) -> jax.Array:
    """Compute normalized Mean Squared Error loss for parameter estimation."""
    pred_theta_norm = model(x_in)
    return jnp.mean((pred_theta_norm - theta_norm_target) ** 2)


@nnx.jit
def stage1_train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    x_in: jax.Array,
    theta_norm_target: jax.Array,
) -> jax.Array:
    """Single JIT-compiled optimization step for Stage 1."""
    grad_fn = nnx.value_and_grad(stage1_loss_fn)
    loss, grads = grad_fn(model, x_in, theta_norm_target)
    optimizer.update(model, grads)
    return loss


def evaluate_stage1(
    model: nnx.Module,
    val_data: dict[str, np.ndarray],
    batch_size: int = 16,
) -> dict[str, float]:
    """Evaluate Stage 1 parameter estimator against validation dataset."""
    rx_full = val_data["rx_full"]
    theta_true = val_data["params"]
    num_samples = len(rx_full)
    all_preds = []

    @nnx.jit
    def predict_batch(model: nnx.Module, x_batch: jax.Array) -> jax.Array:
        u_pred = model(x_batch)
        return LOWER_BOUNDS + u_pred * (UPPER_BOUNDS - LOWER_BOUNDS)

    for i in range(0, num_samples, batch_size):
        batch_rx = rx_full[i : i + batch_size]
        x_in = jax.vmap(radio_idp.extract_features)(jnp.array(batch_rx))
        pred_theta = predict_batch(model, x_in)
        all_preds.append(np.array(pred_theta))

    pred_all = np.concatenate(all_preds, axis=0)
    return compute_param_metrics_dict(
        theta_true=theta_true[: len(pred_all)],
        theta_pred=pred_all,
        param_names=PARAM_NAMES,
        lower_bounds=LOWER_BOUNDS,
        upper_bounds=UPPER_BOUNDS,
    )


def train_stage1(
    model: nnx.Module,
    config: Stage1TrainConfig | None = None,
) -> dict[str, Any]:
    """Train a Stage 1 Parameter Estimator model with Flax NNX.

    Args:
        model: Any nnx.Module accepting input of shape (B, L, 2) and outputting (B, 11).
        config: Optional configuration dataclass.

    Returns:
        Dictionary containing trained model, best validation NRMSE, and run directory.
    """
    if config is None:
        config = Stage1TrainConfig()

    logger = ExperimentLogger(
        log_dir=config.log_dir,
        experiment_name=config.experiment_name,
    )
    checkpoint_mgr = ModelCheckpointManager(
        directory=config.checkpoint_dir,
        max_to_keep=3,
    )

    # Optax schedule and optimizer
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

    # Setup on-the-fly streaming generator configurations
    constellation = radio_idp.get_qam_constellation(4)
    m_power = int(np.log2(config.sobol_points))
    sobol_table = init_sobol_table(m_power=m_power, seed=config.seed)
    
    graphdef, state = nnx.split((model, optimizer))
    
    @jax.jit
    def train_chunk(state, key, key_idx):
        def step_fn(carry, step_key):
            state_val, k_idx = carry
            k_batch, k_idx_new = jax.random.split(k_idx)
            
            indices = jax.random.randint(
                k_batch, shape=(config.batch_size,), minval=0, maxval=config.sobol_points
            )
            k_sobol, k_bursts = jax.random.split(step_key)
            theta = sample_parameters_sobol(k_sobol, sobol_table, indices)
            
            batch_keys = jax.random.split(k_bursts, config.batch_size)
            _, _, rx_full, _, _ = jax.vmap(
                generate_single_burst, in_axes=(0, 0, None, None)
            )(batch_keys, theta, constellation, config.sample_len)
            
            x_in = jax.vmap(radio_idp.extract_features)(rx_full)
            target = (theta - LOWER_BOUNDS) / (UPPER_BOUNDS - LOWER_BOUNDS)
            
            m, opt = nnx.merge(graphdef, state_val)
            def loss_fn(m_local):
                pred = m_local(x_in)
                return jnp.mean((pred - target) ** 2)
                
            loss, grads = nnx.value_and_grad(loss_fn)(m)
            opt.update(m, grads)
            
            _, new_state = nnx.split((m, opt))
            return (new_state, k_idx_new), loss
            
        keys = jax.random.split(key, 50)
        (final_state, final_k_idx), losses = jax.lax.scan(step_fn, (state, key_idx), keys)
        return final_state, final_k_idx, jnp.mean(losses)

    # Load validation data slice
    val_data = load_validation_bursts(config.val_file, max_bursts=config.val_bursts)

    best_nrmse = float("inf")
    pbar = tqdm(range(0, config.num_steps, 50), desc="Stage 1 Training (Chunks)")
    
    key = jax.random.key(config.seed)
    key_idx = jax.random.key(config.seed + 1)

    for step_start in pbar:
        key, subkey = jax.random.split(key)
        state, key_idx, avg_loss = train_chunk(state, subkey, key_idx)
        loss_val = float(avg_loss)
        
        current_step = step_start + 50
        
        logger.log_metrics(current_step, {"loss": loss_val}, prefix="train")
        pbar.set_postfix({"train_loss": f"{loss_val:.5f}"})

        if current_step % config.val_interval == 0 or current_step == config.num_steps:
            nnx.update((model, optimizer), state)
            val_metrics = evaluate_stage1(model, val_data)
            logger.log_metrics(current_step, val_metrics, prefix="val")
            nrmse_mean = val_metrics.get("nrmse/mean", float("inf"))

            if nrmse_mean < best_nrmse:
                best_nrmse = nrmse_mean
                checkpoint_mgr.save(
                    current_step,
                    model,
                    opt_state=optimizer,
                    extra_metadata={"val_nrmse": best_nrmse},
                )

            pbar.set_postfix({
                "train_loss": f"{loss_val:.5f}",
                "val_nrmse": f"{nrmse_mean:.4f}",
            })

    checkpoint_mgr.wait_until_finished()
    logger.close()

    return {
        "model": model,
        "best_nrmse": best_nrmse,
        "run_dir": str(logger.run_dir),
    }
