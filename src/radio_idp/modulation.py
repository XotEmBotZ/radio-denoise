from functools import partial
import jax
import jax.numpy as jnp


def generate_data(key: jax.Array, size: int, num_bits: int = 2) -> jax.Array:
    return jax.random.randint(key, shape=(size,), minval=0, maxval=1 << num_bits)


def get_qam_constellation(num_bits: int = 4) -> jax.Array:
    levels_per_axis = 1 << (num_bits // 2)
    axis = 2 * jnp.arange(levels_per_axis, dtype=jnp.float32) - (levels_per_axis - 1)
    constellation = (axis[:, None] + 1j * axis[None, :]).ravel()
    avg_power = jnp.mean(jnp.abs(constellation) ** 2)
    return constellation / jnp.sqrt(avg_power)


def get_qpsk_constellation() -> jax.Array:
    angles = jnp.pi / 4.0 + jnp.arange(4) * (jnp.pi / 2.0)
    return jnp.exp(1j * angles)


def get_psk_constellation(num_bits: int = 2) -> jax.Array:
    m = 1 << num_bits
    angles = jnp.pi / m + 2.0 * jnp.pi * jnp.arange(m) / m
    return jnp.exp(1j * angles)


def get_apsk_constellation(num_bits: int = 4, gamma: float = 2.75) -> jax.Array:
    if num_bits == 2:
        angles = jnp.pi / 4.0 + jnp.arange(4) * (jnp.pi / 2.0)
        pts = jnp.exp(1j * angles)
    elif num_bits == 3:
        r1 = jnp.exp(1j * (jnp.pi / 4.0 + jnp.arange(4) * (jnp.pi / 2.0)))
        r2 = gamma * jnp.exp(1j * (jnp.arange(4) * (jnp.pi / 2.0)))
        pts = jnp.concatenate([r1, r2])
    elif num_bits == 5:
        r1 = jnp.exp(1j * (jnp.pi / 4.0 + jnp.arange(4) * (jnp.pi / 2.0)))
        r2 = gamma * jnp.exp(1j * (jnp.pi / 12.0 + jnp.arange(12) * (jnp.pi / 6.0)))
        r3 = (gamma * 1.8) * jnp.exp(1j * (jnp.arange(16) * (jnp.pi / 8.0)))
        pts = jnp.concatenate([r1, r2, r3])
    elif num_bits == 6:
        r1 = jnp.exp(1j * (jnp.pi / 4.0 + jnp.arange(4) * (jnp.pi / 2.0)))
        r2 = gamma * jnp.exp(1j * (jnp.pi / 12.0 + jnp.arange(12) * (jnp.pi / 6.0)))
        r3 = (gamma * 1.7) * jnp.exp(
            1j * (jnp.pi / 20.0 + jnp.arange(20) * (jnp.pi / 10.0))
        )
        r4 = (gamma * 2.4) * jnp.exp(1j * (jnp.arange(28) * (2.0 * jnp.pi / 28.0)))
        pts = jnp.concatenate([r1, r2, r3, r4])
    else:
        r1 = jnp.exp(1j * (jnp.pi / 4.0 + jnp.arange(4) * (jnp.pi / 2.0)))
        r2 = gamma * jnp.exp(1j * (jnp.pi / 12.0 + jnp.arange(12) * (jnp.pi / 6.0)))
        pts = jnp.concatenate([r1, r2])
    return pts / jnp.sqrt(jnp.mean(jnp.abs(pts) ** 2))


def encode_symbols(data: jax.Array, constellation: jax.Array) -> jax.Array:
    return constellation[data]


def demap_symbols(received_symbols: jax.Array, constellation: jax.Array) -> jax.Array:
    c_power = 0.5 * (constellation.real**2 + constellation.imag**2)
    scores = (
        received_symbols.real[..., None] * constellation.real
        + received_symbols.imag[..., None] * constellation.imag
        - c_power
    )
    return jnp.argmax(scores, axis=-1)
