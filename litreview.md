The primary sources utilized across the literature review and comparative analysis are detailed below, categorized by their methodological focus:

---

### 1. Neural Receiver Architectures, Over-the-Air Testbeds, and Benchmark Implementations

* **Real-Time 5G OAI Neural Receiver Testbed with USRP X410**
* **Repository / Publication:** Ettus Research / National Instruments Application Notes


* **Key Contributions Used:** Real-time software-defined radio (SDR) integration of the DeepRx architecture with OpenAirInterface (OAI); structural model reduction from 700,000 to 47,000 parameters to achieve sub-500 µs inference on NVIDIA RTX 4090 and A100 GPUs via TensorRT under 5G NR Numerology 1 constraints.




* **Site-Specific Finetuning of Neural and Model-Driven Receivers**
* **Publication:** arXiv:2609.04004


* **Key Contributions Used:** Real-world over-the-air measurement campaigns across indoor laboratory and office floor environments; validation of the 2,500-parameter Model-Driven Receiver (MDX) yielding 1.5 dB to 3.5 dB effective SNR improvements via site-specific fine-tuning.




* **DeepRx-OFDM PyTorch Implementation and Benchmark**
* **Repository:** G-ALI007 / GitHub


* **Key Contributions Used:** Fully convolutional Preactivation ResNet and depthwise separable convolutional baseline; performance evaluations over TDL-B fading channels showing a 1.5x BER reduction over classical LMMSE equalizers under mobility Doppler shifts up to 500 Hz.




* **TransRx: Deep Transformer-Based Neural Receiver for 6G-V2X Communications**
* **Publication:** arXiv:2408.01145


* **Key Contributions Used:** Multi-head self-attention architecture (4 encoder blocks, 128 embedding dimension) evaluated in 28 GHz mmWave V2X scenarios; achieved low-BER convergence at 6.25 dB SNR (a 3.5 dB improvement over CNN receivers and 8 dB to 11 dB over Least Squares estimators), requiring 12 million training samples.





---

### 2. Generative Inverse Solvers, Flow Matching, and Diffusion Models

* **Recursive Flow (RC-Flow): A Closed-Loop Flow-Matching Solver for CSI Recovery**
* **Publication:** arXiv:2601.15767


* **Key Contributions Used:** Closed-loop generative inverse solver uniting flow-matching priors with closed-form physics-aware proximal projections; proof of global asymptotic stability via Jacobian spectral radius bounding ($\rho(\mathbf{J}) < 1$), eliminating low-SNR generative hallucinations.




* **Flow Matching-Based Plug-and-Play Channel Estimation for MIMO Systems**
* **Publication:** arXiv:2601.13827


* **Key Contributions Used:** Plug-and-Play Proximal Gradient Descent (PnP-PGD) framework with continuous velocity fields; achieves up to 49x faster inference and a 20x reduction in peak GPU memory usage compared to iterative score-based diffusion models.





---

### 3. Physics-Informed ML and Hardware Impairment Modeling

* **Causal Representation Learning for Joint Mitigation of Coupled RF Impairments in MIMO Systems**
* **Publication:** *MDPI Electronics*, Vol. 15, No. 6


* **Key Contributions Used:** Physics-informed structural causal model (SCM) inside a Causal Variational Autoencoder (C-VAE); latent disentanglement of power amplifier non-linearities, oscillator phase noise, and thermal noise; demonstrated up to a 2.2 dB output SNR improvement and a 0.86 out-of-distribution robustness score.




* **Intelligent AutoEncoder-Based Modulation: Optimizing Transmission Performance over Non-Ideal Wireless Channels**
* **Publication:** *REV Journal on Electronics and Communications*, Vol. 16, No. 2


* **Key Contributions Used:** End-to-end autoencoders under Rapp PA saturation ($A_{\text{sat}} = 1.2$, $p = 3$, $\alpha_{\text{PM}} = 0.08$) and carrier frequency offset ($\Delta f \cdot T_s = 0.03$); demonstrated non-uniform geometric constellation warping, reducing PAPR from 2.6 dB to 2.3 dB and achieving a 4x BER reduction at 34 dB SNR.




* **Deep Unfolding Neural Networks for Real-Time Sensing and Processing**
* **Publication:** alphAxiv / arXiv:2502.05952v4


* **Key Contributions Used:** Systematic methodology for mapping classical iterative optimization steps (AMP, PGD, ADMM) into deep neural layers, achieving sub-100 µs inference latencies suitable for direct hardware synthesis on ASICs and FPGAs.





---

### 4. Continuous Wave Propagation and RF Hardware Constraints

* **ReVeal: Physics-Informed Neural Networks for High-Accuracy Radio Environment Mapping**
* **Publication:** ARA Rural Wireless Living Lab / IEEE DySPAN


* **Key Contributions Used:** Embedding partial differential equations (PDEs) of spatial radio wave power flow directly into neural loss functions; achieved a 1.95 dB RMSE in signal power cartography across 514 km² using only 30 physical measurement points.




* **Photon Splatting: A Physics-Guided Neural Surrogate for Real-Time Channel Modeling**
* **Publication:** *IEEE Journal of Selected Topics in Electromagnetics Antennas and Propagation* / ResearchGate


* **Key Contributions Used:** Integration of Maxwell-consistent wave scattering physics with neural operator learning for real-time 3D Channel Impulse Response (CIR) estimation.




* **Circuits, Systems, and Architectural Scaling in Post-Moore Photonic/RF Computing**
* **Publication:** arXiv:2604.10841


* **Key Contributions Used:** Analysis of the "precision wall" and analog-to-digital conversion bottlenecks in high-frequency, wideband signal processing hardware.
