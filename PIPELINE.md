# Radio-IDP: Training & Benchmarking Pipeline Architecture

This document specifies the unified architecture, dataflow, and design decisions for the `radio-IDP` training, validation, and evaluation pipeline.

---

## 1. Record of User Decisions & Design Rationale

During the architecture interview, the following explicit engineering decisions were established:

| # | Decision Topic | User Decision | Rationale & Pipeline Impact |
|---|---|---|---|
| **1** | **Python Environment** | Use `uv` strictly | All script executions, package dependencies, and environments must be invoked via `uv run`. |
| **2** | **Training Data Ingestion** | **On-the-Fly Synthetic Generation** (JAX/GPU) | Generates dynamic bursts in VRAM per step; provides infinite data diversity without disk storage constraints or PCIe bottlenecks. |
| **3** | **Validation Data Ingestion** | **Fixed Validation-Specific Datasets** | Pre-generated 65k-burst NetCDF datasets in `data/validation/*.nc` are preserved for repeatable, benchmarked evaluation. |
| **4** | **Loss Function & Model Scope** | **Deferred / Model-Agnostic** | The pipeline must not hardcode a single model or loss function; it must remain generic and flexible so multiple models can be trained and compared side-by-side. |
| **5** | **Neural Network Framework** | **Flax NNX + Optax** | Modern object-oriented Flax module paradigm (`nnx.Module`) backed by official Google/DeepMind tooling and native Orbax integration. |
| **6** | **Model Architecture Organization** | **Separate Flax Classes via Reusable Python API** | Rather than external configuration frameworks (like Hydra), models are defined as clean, standalone Flax classes and passed directly to a reusable Python function: `train(model, ...)`. |
| **7** | **Hybrid Impairment Strategy** | **Two-Stage Decoupled Pipeline** | Separates the task into two independent training stages:<br>• **Stage 1:** Deterministic parameter estimation ($\hat{\theta}$) + physics inversion.<br>• **Stage 2:** Stochastic residual denoising (Flow Matching / subtraction for AWGN + phase noise). |
| **8** | **Stage 2 Training Coupling** | **Parameter Perturbation (Domain Randomization)** | Stage 2 trains on bursts inverted using noisy parameters ($\theta_{\text{noisy}} = \theta_{\text{true}} + \epsilon$). This builds resilience to real-world estimation errors without requiring a live dependency on a trained Stage 1 checkpoint. |
| **9** | **Validation & Evaluation Protocol** | **Two-Tier Evaluation** | • **Tier 1 (In-training):** Fast JIT loss & EVM calculation for progress monitoring and checkpoint selection.<br>• **Tier 2 (Post-training):** Dedicated offline benchmark tool computing publication-grade BER vs. SNR and EVM vs. SNR waterfall curves across all modulations. |
| **10** | **Logging & Persistence** | **TensorBoard + Orbax + JSONL** | Offline real-time web visualization via TensorBoard, standard Flax/Orbax PyTree checkpointing, and structured local JSONL run logs. |

---

## 2. High-Level System Architecture

The pipeline implements the **Two-Stage Decoupled Framework** agreed upon:

```
                            +-----------------------------------------------+
                            |              RAW RECEIVED BURST               |
                            |           y in C^{1024} (rx_full)             |
                            +-----------------------------------------------+
                                                    |
                                                    v
+-------------------------------------------------------------------------------------------------------+
| STAGE 1: DETERMINISTIC PARAMETER ESTIMATOR                                                            |
|                                                                                                       |
| Neural Network: y -> theta_hat in R^{11}                                                              |
|   Estimates: CFO, phase noise std, IQ amplitude/phase imbalance, Rapp p & V_sat,                      |
|              DC offset (dc_i, dc_q), Multipath decay & K-factor                                       |
|                                                                                                       |
| Physics Inversion Block:                                                                              |
|   y_cleaned = deterministic_denoise_pipeline(y, theta_hat)                                            |
|   (Inverts DC offset -> CFO -> IQ imbalance -> Rapp HPA -> Multipath delay spread)                   |
+-------------------------------------------------------------------------------------------------------+
                                                    |
                                                    v
+-------------------------------------------------------------------------------------------------------+
| STAGE 2: STOCHASTIC FLOW DENOISER                                                                     |
|                                                                                                       |
| Neural Network: (x_t, t, y_cleaned) -> v_t in R^{1024 x 2}                                            |
|   Learns continuous velocity field to transport stochastic residuals (AWGN + Wiener phase noise)      |
|   to the clean transmitted constellation symbols x_tx                                                 |
|                                                                                                       |
| Inference Solver:                                                                                     |
|   Integrates ODE dx/dt = v_theta(x, t, y_cleaned) from t=0 to t=1 (Euler / RK4 / Heun)               |
+-------------------------------------------------------------------------------------------------------+
                                                    |
                                                    v
+-------------------------------------------------------------------------------------------------------+
| EVALUATION & DEMAPPING                                                                                |
|   Recovered symbols x_hat -> Nearest-Neighbor Constellation Demapper -> Predicted Labels / Bits       |
|   Metrics: EVM (dB), Symbol Error Rate (SER), Bit Error Rate (BER)                                    |
+-------------------------------------------------------------------------------------------------------+
```

---

## 3. Detailed Component Specifications

### 3.1. Data Ingestion & Streaming Strategy
* **Training Data:** **On-the-Fly Synthetic Streaming (Pure JAX/GPU).**
  * Training batches are synthesized dynamically using vectorized JAX primitives (`radio_idp.channel`, `radio_idp.modulation`).
  * Infinite sample diversity, zero overfitting to static bursts, zero disk storage consumption, and zero PCIe host-to-device I/O bottlenecks.
  * Parameters $\theta$ are continuously drawn across predefined operational bounds via Sobol pseudo-random or uniform sampling.
* **Validation Data:** **Fixed Pre-Generated Datasets.**
  * Sourced from `data/validation/val_sobol_{modulation}_32k.nc` across 6 modulation schemes:
    * `qpsk`, `8psk`, `16qam`, `64qam`, `16apsk`, `32apsk`.
  * Dimensions: `(burst: 65536, symbol: 1024, param: 11)`.
  * Guarantees strict benchmark repeatability across all model iterations and baseline comparisons.

### 3.2. Stage 2 Training Ingestion & Coupling
* **Coupling Mechanism:** **Parameter Perturbation (Domain Randomization).**
  * Rather than strictly depending on a live Stage 1 checkpoint or assuming an unrealistic perfect oracle, the on-the-fly streaming pipeline inverts bursts with perturbed parameters:
    $$\theta_{\text{noisy}} = \theta_{\text{true}} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_\theta^2)$$
  * Allows Stage 2 models to be trained independently while enforcing robustness to residual phase, frequency, and non-linear distortion errors from Stage 1.
  * An optional flag (`--stage1-checkpoint`) enables fine-tuning Stage 2 directly with a frozen Stage 1 estimator.

### 3.3. Neural Framework & Optimization Stack
* **Framework:** **Flax NNX + Optax** running natively on **JAX** (with CUDA CTK 13+ support).
* **Package Management:** Managed strictly via `uv` (`uv run ...`).
* **Design Paradigm:** Object-oriented `nnx.Module` classes with pure PyTree state management and Optax gradient transformations (`optax.adamw`, cosine decay schedules, gradient clipping).

### 3.4. Uniform Model Interface Protocol
The pipeline is **model-agnostic**, exposing reusable Python training functions:
* `train_stage1(model: nnx.Module, config: TrainConfig)`
* `train_stage2(model: nnx.Module, config: TrainConfig)`

Any Flax module can be plugged into the pipeline as long as it adheres to the standard call signatures:
* **Stage 1 Models (Parameter Estimators):**
  ```python
  theta_hat = model(y)  # y: (batch, 1024, 2) [Re, Im] -> theta_hat: (batch, 11)
  ```
* **Stage 2 Models (Flow Denoisers):**
  ```python
  # x_cat stacks the flow state x_t and the inverted condition y_cleaned along the feature channel
  # Shape: (batch, 1024, 4) -> v_t: (batch, 1024, 2)
  v_t = model(x_cat, t)
  ```

### 3.5. Validation & Evaluation Protocol (Two-Tier)
1. **Tier 1 (In-Training Fast Validation):**
   * Periodically evaluated during the training loop on a validation subset.
   * **Stage 1 Metric:** Normalized Root Mean Squared Error (NRMSE) across the 11 physical parameters.
   * **Stage 2 Metric:** Validation Flow Matching Loss and Error Vector Magnitude (EVM in dB):
     $$\text{EVM (dB)} = 10 \log_{10} \left( \frac{\mathbb{E}[|x_{\text{tx}} - \hat{x}|^2]}{\mathbb{E}[|x_{\text{tx}}|^2]} \right)$$
   * Used for early stopping and saving `best_checkpoint`.
2. **Tier 2 (Post-Training Benchmarking Suite):**
   * Standalone evaluation runner operating over the full 65k bursts in `data/validation/*.nc`.
   * Evaluates across SNR bins (e.g. $-5\text{ dB}$ to $+30\text{ dB}$ in $5\text{ dB}$ steps).
   * Computes:
     * **BER vs. SNR** and **SER vs. SNR** waterfall curves.
     * **EVM vs. SNR** curves.
     * Runtime inference latency (µs per burst) and ODE function evaluation counts (NFE).

### 3.6. Logging & Checkpoint Management
* **Live Experiment Monitoring:** **TensorBoard** (`runs/` directory) for real-time loss, EVM, learning rate, and parameter convergence curves.
* **Structured Local History:** JSONL/CSV metric logs generated per run for external script analysis and Matplotlib figure generation.
* **Checkpoint Serialization:** **Orbax Checkpoint Manager** saving model weights, optimizer state, and training metadata with top-$K$ retention.

---

## 4. Directory Layout

```
radio-IDP/
├── src/
│   └── radio_idp/
│       ├── channel.py             # JAX physical impairment forward models
│       ├── modulation.py          # Constellations and bit mapping/demapping
│       ├── inversion.py           # Differentiable deterministic physics inverters
│       ├── dataset.py             # JAX on-the-fly streaming generator & NetCDF loader
│       ├── models/
│       │   ├── stage1/            # Stage 1 parameter estimator architectures
│       │   └── stage2/            # Stage 2 flow matching denoiser backbones (UNet, ResNet, etc.)
│       ├── train_stage1.py        # Stage 1 training runner
│       ├── train_stage2.py        # Stage 2 training runner
│       ├── benchmark.py           # Tier-2 BER/SER/EVM waterfall evaluation tool
│       └── utils/
│           ├── metrics.py         # EVM, BER, SER calculation utilities
│           └── checkpoint.py      # Orbax checkpoint wrappers
├── data/
│   └── validation/                # Pre-generated 65k-burst NetCDF validation sets
├── notebooks/                     # Interactive exploration & visualization dashboards
├── configs/                       # Configuration dataclasses & run presets
├── PIPELINE.md                    # This architecture document
└── pyproject.toml                 # Project metadata & dependencies (uv managed)
```
