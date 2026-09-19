Here is the architectural blueprint of a digital communications simulation pipeline, mapping out what is already built, what modules come next, and how QAM and APSK diverge across the chain.

---

### End-to-End Simulation Architecture

```text
[Bit Generator] 
       │
[FEC Encoder / Interleaver] 
       │
[Bit-to-Symbol Mapper]  ◄─── (Modulation Dependent: QAM Grid vs. APSK Rings)
       │
[Pulse Shaping (RRC)]   ─── Upsampling to discrete baseband waveform
       │
[TX Front-End Distortions] ─── IQ Imbalance, DAC Quantization, PA Saturation (AM/AM, AM/PM)
       │
═══════╡ CHANNEL ╞═══════════════════════════════════════════════════════════════════
       │ (Multipath / Delay Spread, Doppler Shift, Carrier Frequency Offset, AWGN)
═════════════════════════════════════════════════════════════════════════════════════
       │
[RX Front-End (AGC & ADC)]
       │
[Carrier & Phase Sync (PLL / Costas)] ─── De-rotates frequency offset (CFO)
       │
[Matched Filtering (RRC)]
       │
[Timing Recovery (Gardner / TED)]     ─── Optimal symbol downsampling
       │
[Equalizer (LMS / RLS / ZF)]          ─── Inverts multipath fading
       │
[Demapper / Slicer]     ◄─── (Modulation Dependent: Cartesian vs. Polar Boundaries)
       │
[FEC Decoder]
       │
[BER / SER Calculation]

```

---

### Pipeline Breakdown & Status

| Stage | Sub-Block | Status | Function in Simulation |
| --- | --- | --- | --- |
| **Data Generation** | PRBS / Random Bit Generator | **Built** | Generates raw entropy (`0`s and `1`s or integer symbol states). |
|  | Forward Error Correction (LDPC/BCH) | *Pending* | Adds parity bits to enable error detection and correction. |
| **Modulation** | Constellation Builder | **Built** (QAM, APSK, & QPSK/PSK) | Generates normalized reference coordinates in the complex plane. |
|  | Symbol Mapper | **Built** | Maps bit vectors or integer indices to complex coordinates $I + jQ$. |
| **Waveform Shaping** | Upsampling & RRC Filter | *Pending* | Converts discrete impulses into a bandwidth-limited baseband waveform. |
| **TX Impairments** | High Power Amplifier (HPA) | **Built** | Non-linear Rapp saturation model ($r_{\text{out}} = r / (1 + (r/V_{\text{sat}})^{2p})^{1/2p}$). |
|  | IQ Imbalance | **Built** | Simulates mixer amplitude stretch and phase shear ($Q' = g(Q\cos\phi + I\sin\phi)$). |
|  | DC Offset (LO Leakage) | **Built** | Additive direct current leakage from local oscillator ($s + (I_{\text{dc}} + jQ_{\text{dc}})$). |
|  | DAC Quantization | **Built** | Finite bit-depth uniform mid-tread quantization on TX analog output. |
| **Channel** | Carrier Frequency Offset (CFO) & Phase Noise | **Built** | Models oscillator mismatch ($e^{j 2\pi \Delta f_{\text{norm}} n}$) and Wiener random-walk phase jitter. |
|  | Multipath Fading | **Built** | FIR multi-tap delay spread filter with Rayleigh / Rician fading (`mode='same'`). |
|  | AWGN Injection | **Built** | Injects complex circular Gaussian noise based on target $E_b/N_0$ or SNR. |
| **RX Front-End** | ADC Quantization | **Built** | Finite bit-depth uniform quantization on incoming RX baseband signal. |
| **RX Synchronization** | Carrier Recovery (PLL) | *Pending* | Tracks and strips out CFO and residual phase drift. |
|  | Symbol Timing Recovery | *Pending* | Locks onto the ideal eye-diagram opening to downsample back to 1 sample/symbol. |
|  | Adaptive Channel Equalizer | *Pending* | Inverts channel multi-tap distortion using algorithms like LMS. |
| **Demodulation** | Euclidean Distance Slicer | **Built** | Classifies noisy received symbols into nearest constellation points. |
| **Verification** | SER Benchmark | **Built** | Simulates and benchmarks error rates across SNR ranges (JAX-accelerated). |

---

### What Is Common vs. What Differs: QAM vs. APSK

**What is 100% Identical:**

* **Bit generation & FEC:** Channel coding schemes (such as DVB-S2 LDPC/BCH frames) work uniformly regardless of the constellation.
* **Baseband Pulse Shaping:** Both rely on Root-Raised Cosine (RRC) filtering with a roll-off factor $\alpha$ to eliminate Inter-Symbol Interference (ISI).
* **Linear Channel Physics:** AWGN generation, Doppler shift, multipath fading channels, and timing recovery (Gardner loops) operate on generic complex samples $I + jQ$ without caring about constellation geometry.

**Where QAM and APSK Diverge:**

* **Constellation Geometry:**
* **QAM:** Cartesian grid ($I$ and $Q$ components are decoupled independent PAM signals).
* **APSK:** Polar layout (concentric rings of PSK constellations; $I$ and $Q$ are strictly coupled via radius and phase angle).


* **Peak-to-Average Power Ratio (PAPR) & Amplifier Tolerance:**
* **QAM:** High PAPR. Outer corner constellation points carry significantly more energy than center points, causing clipping and spectral regrowth in non-linear amplifiers. Requires high power back-off (inefficient).
* **APSK:** Very low PAPR. Points sit on 2 or 3 circular rings of constant amplitude. Highly resilient to non-linear high-power amplifiers (HPAs) operated near saturation (which is why satellites use 16/32-APSK instead of 16/64-QAM).


* **Demapping & Decision Boundaries:**
* **QAM:** Simple independent thresholding on $I$ and $Q$ axes ($O(1)$ decision complexity).
* **APSK:** Coupled circular/radial boundaries, requiring true 2D minimum Euclidean distance or polar coordinate transformation ($r, \theta$).


* **Carrier Phase Synchronization:**
* **QAM:** Standard Costas loop (exploits $90^\circ$ four-fold symmetry).
* **APSK:** Requires decision-directed loops or pilot-symbol assisted recovery because inner and outer rings typically have different rotational symmetries (e.g., 4 inner points vs. 12 outer points).



---

### Flow Matching / Diffusion Conditioning & Overall SNR Formulation

To train a Flow Matching or Diffusion model on synthetic radio waveforms where time $t \in [0, 1]$ parameterizes signal degradation, two complementary approaches are defined:

#### Approach 1: Effective SNR / SINAD (EVM-Based Ground-Truth Metric)
Quantifies the total aggregate distortion across all combined channel impairments (AWGN, CFO, phase noise, IQ imbalance, clipping):
$$e[n] = r[n] - s[n]$$
$$P_{\text{signal}} = \mathbb{E}[|s[n]|^2], \quad P_{\text{error}} = \mathbb{E}[|r[n] - s[n]|^2]$$
$$\text{SNR}_{\text{eff, dB}} = 10 \log_{10} \left( \frac{P_{\text{signal}}}{P_{\text{error}}} \right) = -20 \log_{10}(\text{EVM})$$
* **Behavior:** When only AWGN is present, $\text{SNR}_{\text{eff, dB}} = \text{SNR}_{\text{AWGN, dB}}$. As phase noise, CFO, clipping, or IQ imbalance are introduced, the error vector power grows and $\text{SNR}_{\text{eff, dB}}$ drops naturally, providing a continuous scalar rating for conditional generation.

#### Approach 2: Generative Degradation Schedule ($t \in [0, 1]$)
Couples each physical impairment severity directly to the continuous Flow time parameter $t$:
* $t = 0$ (Clean Target): $\text{SNR} = 35\text{ dB}$, $\Delta f_{\text{norm}} = 0$, $\sigma_{\theta} = 0$, $g = 1.0$, $\phi = 0$.
* $t = 1$ (Fully Degraded Base): $\text{SNR} = -5\text{ dB}$, $\Delta f_{\text{norm}} = 0.05$, $\sigma_{\theta} = 0.1\text{ rad}$, $g = 1.3$, $\phi = 10^\circ$.
* Monotonic parameter interpolation: $\text{param}(t) = \text{param}_{\text{clean}} + t \cdot (\text{param}_{\text{degraded}} - \text{param}_{\text{clean}})$.

---

### Channel Impairments Status & Remaining Pipeline

1. **AWGN (Thermal Noise):** **Built**
2. **Carrier Frequency Offset (CFO):** **Built**
3. **Phase Noise (Wiener Process):** **Built**
4. **IQ Imbalance (Gain & Phase Mismatch):** **Built**
5. **High Power Amplifier (HPA) Non-Linearity (Rapp Model):** **Built**
6. **DC Offset (LO Leakage):** **Built**
7. **Multipath Fading (Rayleigh / Rician Delay Spread):** **Built**
8. **ADC/DAC Quantization Noise:** **Built**

---

### Hybrid Machine Learning Architecture: Physics Estimator + Flow Matching

To solve the dual challenges of physical channel distortion and high-entropy stochastic thermal noise without rotational ambiguity or excessive generative sampling, a **Two-Stage Hybrid Architecture** is adopted:

```text
Received Waveform r[n] (All Impairments)
       │
═══════╪═════════════════════════════════════════════════════════════════
STAGE 1: DETERMINISTIC PARAMETER ESTIMATION & ANALYTICAL INVERSION
═══════╪═════════════════════════════════════════════════════════════════
       │
       ├──────────────────────────────────────┐
       ▼                                      │
┌──────────────────────────────┐              │
│ Neural Estimator (CNN / MLP) │              │
│ Predicts physical scalars:   │              │
│ • CFO (\hat{\Delta f})       │              │
│ • IQ Imbalance (\hat{g}, \hat{\phi})        │
│ • DC Offset (\hat{I}, \hat{Q})              │
│ • Multipath Taps (\hat{\mathbf{h}})         │
│ • Effective SNR (\hat{SNR})  │              │
└──────────────┬───────────────┘              │
               │ Estimated \hat{\theta}       │
               ▼                              ▼
      ┌──────────────────────────────────────────┐
      │ Exact Analytical Inverse DSP (in JAX)    │
      │ 1. De-rotate CFO: e^{-j 2\pi \hat{f} n}  │
      │ 2. Cancel DC Offset: r - \hat{DC}        │
      │ 3. Invert IQ Imbalance: [2x2 Matrix]^{-1}│
      │ 4. Equalize Multipath: ZF / MMSE         │
      │ 5. HPA Post-Distortion                   │
      └──────────────────┬───────────────────────┘
                         │
                         ▼
       Pre-Aligned, Centered & De-Rotated Waveform r_{aligned}[n]
       (Constellation geometry is restored; only stochastic noise remains)
                         │
═════════════════════════╪═══════════════════════════════════════════════
STAGE 2: STOCHASTIC GENERATIVE DENOISING (Conditional Flow Matching)
═══════════════════════════════════════════════════════════════════════
                         │
                         │ Conditioned on t \propto 1 / \hat{SNR}_{eff}
                         ▼
      ┌──────────────────────────────────────────┐
      │ Continuous Normalizing Flow              │
      │ (Neural ODE Vector Field v_t)            │
      │                                          │
      │ • Contracts Gaussian thermal AWGN clouds │
      │ • Compensates residual Wiener phase drift│
      │ • Fast sampling: Converges in 2-4 Euler  │
      │   steps due to pre-aligned geometry      │
      └──────────────────┬───────────────────────┘
                         │
                         ▼
              Clean Reconstructed Symbols s[n]
```

#### Breakdown by Impairment:

| Impairment | Processing Stage | Method | Mathematical Formulation |
| :--- | :---: | :--- | :--- |
| **DC Offset** | **Stage 1** | Analytical Subtraction | $r[n] - (\hat{I}_{\text{dc}} + j\hat{Q}_{\text{dc}})$ |
| **CFO** | **Stage 1** | Analytical De-rotation | $r[n] \cdot e^{-j 2\pi \hat{f} n}$ |
| **IQ Imbalance** | **Stage 1** | $2 \times 2$ Matrix Inverse | $[I, Q]^T \leftarrow \begin{bmatrix} 1 & 0 \\ -\tan\hat{\phi} & \frac{1}{\hat{g}\cos\hat{\phi}} \end{bmatrix} [I, Q]^T$ |
| **Multipath Fading** | **Stage 1** | Frequency-Domain Equalizer | $S(f) = R(f) / \hat{H}(f)$ (Zero-Forcing / MMSE) |
| **HPA Rapp Saturation** | **Stage 1** | Digital Post-Distortion | Inverts AM/AM compression curve below $V_{\text{sat}}$ |
| **AWGN Thermal Noise** | **Stage 2** | **Conditional Flow Matching** | Probability flow ODE collapsing Gaussian cloud into point mixture |
| **Phase Noise (Wiener)**| **Stage 2** | **Flow Matching Trajectory** | Transports tangential phase jitter back to radial lattice |
| **Quantization Noise** | **Stage 2** | **Flow Matching Regularization**| Reconstructs sub-quantization continuous amplitude |

---

### Next Milestones
1. **Stage 1 Analytical Inverters (`deterministic_denoise_pipeline`):** **Built** (Inverts CFO, Multipath, HPA, DC Offset, and IQ Imbalance in exact reverse physical order).
2. **Build Effective SNR / EVM Evaluator:** Implement `compute_effective_snr_db(tx, rx)` to quantify total degradation and supply conditioning parameter $t$.
3. **Stage 2 Flow Matching Core:** Set up the continuous vector field training pipeline in JAX/Equinox/Flax.