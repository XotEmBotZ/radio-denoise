from functools import partial
import jax
import jax.numpy as jnp


def add_awgn(
    symbols: jax.Array, snr_db: float | jax.Array, key: jax.Array
) -> jax.Array:
    snr_linear = 10.0 ** (snr_db / 10.0)
    signal_power = jnp.mean(jnp.abs(symbols) ** 2)
    noise_power = signal_power / snr_linear
    noise_per_axis = jnp.sqrt(noise_power / 2.0)
    key_r, key_i = jax.random.split(key)
    noise = noise_per_axis * (
        jax.random.normal(key_r, shape=symbols.shape)
        + 1j * jax.random.normal(key_i, shape=symbols.shape)
    )
    return symbols + noise


def add_cfo(symbols: jax.Array, cfo_norm: float | jax.Array) -> jax.Array:
    theta = 2.0 * jnp.pi * cfo_norm * jnp.arange(symbols.shape[0])
    phasor = jnp.exp(1j * theta)
    return symbols * phasor


def add_phase_noise(
    symbols: jax.Array, phase_noise_std: float | jax.Array, key: jax.Array
) -> jax.Array:
    noise = jax.random.normal(key, shape=symbols.shape) * phase_noise_std
    cum_noise = jnp.cumsum(noise)
    phasor = jnp.exp(1j * cum_noise)
    return symbols * phasor


def add_iq_imbalance(
    symbols: jax.Array,
    amp_imb: float | jax.Array,
    phase_imb_rad: float | jax.Array,
) -> jax.Array:
    I = symbols.real
    Q = symbols.imag
    Q = amp_imb * (Q * jnp.cos(phase_imb_rad) + I * jnp.sin(phase_imb_rad))
    return I + 1j * Q


def add_hpa_rapp(
    symbols: jax.Array,
    v_sat: float | jax.Array = 0.9,
    p: float | jax.Array = 2.0,
) -> jax.Array:
    gain_compression = (1.0 + (jnp.abs(symbols) / v_sat) ** (2.0 * p)) ** (
        1.0 / (2.0 * p)
    )
    return symbols / gain_compression


def add_dc_offset(
    symbols: jax.Array, dc_i: float | jax.Array, dc_q: float | jax.Array
) -> jax.Array:
    return symbols + (dc_i + 1j * dc_q)


def get_multipath_taps(
    key: jax.Array,
    num_taps: int = 3,
    decay: float | jax.Array = 0.5,
    k_factor: float | jax.Array = 0.0,
) -> jax.Array:
    delays = jnp.arange(num_taps)
    pdp = jnp.exp(-decay * delays)
    pdp = pdp / jnp.sum(pdp)
    key_r, key_i = jax.random.split(key)
    diffuse = (
        jax.random.normal(key_r, (num_taps,))
        + 1j * jax.random.normal(key_i, (num_taps,))
    ) / jnp.sqrt(2.0)
    los = jnp.zeros(num_taps, dtype=jnp.complex64).at[0].set(1.0 + 0.0j)
    taps = (
        jnp.sqrt(k_factor / (k_factor + 1.0)) * los
        + jnp.sqrt(1.0 / (k_factor + 1.0)) * diffuse
    )
    taps = taps * jnp.sqrt(pdp)
    return taps / jnp.sqrt(jnp.sum(jnp.abs(taps) ** 2))


def add_multipath_fading(symbols: jax.Array, taps: jax.Array) -> jax.Array:
    return jnp.convolve(symbols, taps, mode="same")


def quantize_uniform(
    signal: jax.Array, num_bits: int = 8, v_max: float | jax.Array = 1.5
) -> jax.Array:
    num_levels = 2**num_bits
    step = (2.0 * v_max) / (num_levels - 1)
    real_q = jnp.clip(jnp.round(signal.real / step) * step, -v_max, v_max)
    imag_q = jnp.clip(jnp.round(signal.imag / step) * step, -v_max, v_max)
    return real_q + 1j * imag_q


def add_dac_quantization(
    symbols: jax.Array, num_bits: int = 10, v_max: float | jax.Array = 1.5
) -> jax.Array:
    return quantize_uniform(symbols, num_bits=num_bits, v_max=v_max)


def add_adc_quantization(
    symbols: jax.Array, num_bits: int = 8, v_max: float | jax.Array = 1.5
) -> jax.Array:
    return quantize_uniform(symbols, num_bits=num_bits, v_max=v_max)


def deterministic_pipeline(
    symbols: jax.Array,
    dac_bits: int = 10,
    v_dac_max: float | jax.Array = 1.5,
    amp_imb: float | jax.Array = 1.0,
    phase_imb_rad: float | jax.Array = 0.0,
    dc_i: float | jax.Array = 0.0,
    dc_q: float | jax.Array = 0.0,
    v_sat: float | jax.Array = 13.0,
    p: float | jax.Array = 2.0,
    taps: jax.Array = jnp.array([1.0], dtype=jnp.complex64),
    cfo_norm: float | jax.Array = 0.0,
) -> jax.Array:
    symbols = add_dac_quantization(symbols, num_bits=dac_bits, v_max=v_dac_max)
    symbols = add_iq_imbalance(symbols, amp_imb=amp_imb, phase_imb_rad=phase_imb_rad)
    symbols = add_dc_offset(symbols, dc_i=dc_i, dc_q=dc_q)
    symbols = add_hpa_rapp(symbols, v_sat=v_sat, p=p)
    symbols = add_multipath_fading(symbols, taps=taps)
    symbols = add_cfo(symbols, cfo_norm=cfo_norm)
    return symbols


def stochastic_pipeline(
    symbols: jax.Array,
    rng: jax.Array,
    phase_noise_std: float | jax.Array = 0.0,
    snr_db: float | jax.Array = 30.0,
    adc_bits: int = 10,
    v_adc_max: float | jax.Array = 1.5,
) -> jax.Array:
    k_pn, k_awgn = jax.random.split(rng, 2)
    symbols = add_phase_noise(symbols, phase_noise_std=phase_noise_std, key=k_pn)
    symbols = add_awgn(symbols, snr_db=snr_db, key=k_awgn)
    symbols = add_adc_quantization(symbols, num_bits=adc_bits, v_max=v_adc_max)
    return symbols


def noise_pipeline(
    symbols: jax.Array,
    rng: jax.Array,
    dac_bits: int = 10,
    v_dac_max: float | jax.Array = 1.5,
    amp_imb: float | jax.Array = 1.0,
    phase_imb_rad: float | jax.Array = 0.1,
    dc_i: float | jax.Array = 0.0,
    dc_q: float | jax.Array = 0.0,
    v_sat: float | jax.Array = 13.0,
    p: float | jax.Array = 2.0,
    taps: jax.Array = jnp.arange(2),
    cfo_norm: float | jax.Array = 2.0,
    phase_noise_std: float | jax.Array = 1.0,
    snr_db: float | jax.Array = 2.0,
    adc_bits: int = 10,
    v_adc_max: float | jax.Array = 1.5,
) -> jax.Array:
    symbols = deterministic_pipeline(
        symbols,
        dac_bits=dac_bits,
        v_dac_max=v_dac_max,
        amp_imb=amp_imb,
        phase_imb_rad=phase_imb_rad,
        dc_i=dc_i,
        dc_q=dc_q,
        v_sat=v_sat,
        p=p,
        taps=taps,
        cfo_norm=cfo_norm,
    )
    symbols = stochastic_pipeline(
        symbols,
        rng=rng,
        phase_noise_std=phase_noise_std,
        snr_db=snr_db,
        adc_bits=adc_bits,
        v_adc_max=v_adc_max,
    )
    return symbols
