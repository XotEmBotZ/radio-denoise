"""Train TransformerStage1Estimator and execute Tier-2 benchmark on completion."""

import os

# Limit JAX VRAM preallocation to 85% to reserve headroom for system/display on RTX 4060
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

import flax.nnx as nnx
import radio_idp
from radio_idp.models import TransformerStage1Estimator


def main() -> None:
    seed = 42

    # 1. Instantiate the Transformer Stage 1 model tuned for 8 GB RTX 4060 GPU
    model = TransformerStage1Estimator(
        in_channels=5,
        d_model=64,
        num_heads=4,
        num_layers=4,
        mlp_dim=128,
        head_hidden_dim=128,
        max_seq_len=1024,
        use_learnable_pe=True,
        rngs=nnx.Rngs(seed),
    )

    # 2. Configure training parameters maximizing GPU utilization without OOM
    train_config = radio_idp.Stage1TrainConfig(
        num_steps=50_000,
        batch_size=16,
        sample_len=1024,
        learning_rate=1e-3,
        warmup_steps=500,
        val_interval=500,
        val_bursts=1024,
        val_file="data/validation/val_sobol_16qam_32k.nc",
        checkpoint_dir="checkpoints/stage1_transformer",
        log_dir="runs",
        experiment_name="stage1_transformer",
        seed=seed,
    )

    print("=" * 60)
    print("Step 1: Training Transformer Stage 1 Parameter Estimator")
    print("=" * 60)
    train_results = radio_idp.train_stage1(model, train_config)

    print("\nTraining completed successfully!")
    print(f"Best Validation NRMSE: {train_results['best_nrmse']:.4f}")
    print(f"Checkpoints saved in: {train_config.checkpoint_dir}")
    print(f"TensorBoard logs saved in: {train_results['run_dir']}")

    # 3. Run full Tier-2 benchmark using trained Stage 1 model
    print("\n" + "=" * 60)
    print("Step 2: Running Full Tier-2 Benchmark Evaluation")
    print("=" * 60)

    benchmark_config = radio_idp.BenchmarkConfig(
        val_dir="data/validation",
        modulations=["16qam"],
        max_bursts_per_mod=4096,
        batch_size=16,
        output_dir="benchmark_results/stage1_transformer",
    )

    bench_results = radio_idp.run_benchmark(
        stage1_model=model,
        config=benchmark_config,
    )

    print("\n" + "=" * 60)
    print("Benchmark Summary Across Modulations:")
    print("=" * 60)
    for mod, data in bench_results.get("modulations", {}).items():
        print(
            f"Modulation: {mod.upper():<7} | "
            f"Overall BER: {data['overall_ber']:.6f} | "
            f"Overall EVM: {data['overall_evm_db']:>6.2f} dB | "
            f"Latency: {data['mean_latency_us']:.1f} µs/burst"
        )

    print(
        f"\nBenchmark waterfall curves and summary JSON saved to: {benchmark_config.output_dir}/"
    )


if __name__ == "__main__":
    main()
