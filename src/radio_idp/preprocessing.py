"""Signal preprocessing and feature extraction for blind parameter estimation."""

import jax
import jax.numpy as jnp

def extract_features(rx_full: jax.Array) -> jax.Array:
    """Extracts 5-channel physical features from raw complex IQ symbols.
    
    Args:
        rx_full: Complex received burst of shape (L,).
        
    Returns:
        Feature tensor of shape (L, 5) containing:
        - [0]: In-phase (I)
        - [1]: Quadrature (Q)
        - [2]: Magnitude
        - [3]: Differential Phase
        - [4]: Normalized Centered FFT Magnitude
    """
    # 1 & 2: I and Q channels
    i_chan = rx_full.real
    q_chan = rx_full.imag
    
    # 3: Magnitude (Envelope)
    mag = jnp.abs(rx_full)
    
    # 4: Differential Phase
    phase = jnp.angle(rx_full)
    diff_phase = jnp.diff(phase, prepend=phase[0])
    # Wrap differential phase to [-pi, pi] to avoid artificial jump artifacts
    diff_phase = (diff_phase + jnp.pi) % (2 * jnp.pi) - jnp.pi
    
    # 5: FFT Magnitude
    # Compute FFT, shift zero frequency to center, and take magnitude
    fft_mag = jnp.abs(jnp.fft.fftshift(jnp.fft.fft(rx_full)))
    # Normalize FFT magnitude by sqrt(L) to preserve energy scale
    fft_mag = fft_mag / jnp.sqrt(rx_full.shape[0])
    
    # Stack features into shape (L, 5)
    return jnp.stack([i_chan, q_chan, mag, diff_phase, fft_mag], axis=-1)
