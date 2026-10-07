import jax
import jax.numpy as jnp


def invert_dc_offset(
    symbols: jax.Array, dc_i: float | jax.Array, dc_q: float | jax.Array
) -> jax.Array:
    return symbols - (dc_i + 1j * dc_q)


def invert_cfo(symbols: jax.Array, cfo_norm: float | jax.Array) -> jax.Array:
    theta = 2.0 * jnp.pi * cfo_norm * jnp.arange(symbols.shape[0])
    return symbols * jnp.exp(-1j * theta)


def invert_iq_imbalance(
    symbols: jax.Array,
    amp_imb: float | jax.Array,
    phase_imb_rad: float | jax.Array,
) -> jax.Array:
    I = symbols.real
    Q = symbols.imag
    Q_clean = (Q / amp_imb - I * jnp.sin(phase_imb_rad)) / jnp.cos(phase_imb_rad)
    return I + 1j * Q_clean


def invert_hpa_rapp(
    symbols: jax.Array,
    v_sat: float | jax.Array = 0.9,
    p: float | jax.Array = 2.0,
) -> jax.Array:
    ratio = jnp.minimum(jnp.abs(symbols) / v_sat, 0.9999)
    gain_expansion = (1.0 - ratio ** (2.0 * p)) ** (1.0 / (2.0 * p))
    return symbols / gain_expansion


def invert_multipath_fading(
    symbols: jax.Array, taps: jax.Array, reg: float | jax.Array = 1e-4
) -> jax.Array:
    n = symbols.shape[0]
    delay = (taps.shape[0] - 1) // 2
    shift = jnp.exp(1j * 2.0 * jnp.pi * delay * jnp.arange(n) / n)
    H = jnp.fft.fft(taps, n=n) * shift
    X_hat = (jnp.fft.fft(symbols) * jnp.conj(H)) / (jnp.abs(H) ** 2 + reg)
    return jnp.fft.ifft(X_hat)


def deterministic_denoise_pipeline(
    symbols: jax.Array,
    cfo_norm: float | jax.Array = 0.0,
    taps: jax.Array = jnp.array([1.0], dtype=jnp.complex64),
    v_sat: float | jax.Array = 13.0,
    p: float | jax.Array = 2.0,
    dc_i: float | jax.Array = 0.0,
    dc_q: float | jax.Array = 0.0,
    amp_imb: float | jax.Array = 1.0,
    phase_imb_rad: float | jax.Array = 0.0,
) -> jax.Array:
    symbols = invert_cfo(symbols, cfo_norm=cfo_norm)
    symbols = invert_multipath_fading(symbols, taps=taps)
    symbols = invert_hpa_rapp(symbols, v_sat=v_sat, p=p)
    symbols = invert_dc_offset(symbols, dc_i=dc_i, dc_q=dc_q)
    symbols = invert_iq_imbalance(
        symbols, amp_imb=amp_imb, phase_imb_rad=phase_imb_rad
    )
    return symbols
