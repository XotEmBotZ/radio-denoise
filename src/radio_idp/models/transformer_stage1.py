"""Transformer + MLP Stage 1 Parameter Estimator model for radio-IDP."""

from dataclasses import dataclass
import flax.nnx as nnx
import jax
import jax.numpy as jnp


def get_sinusoidal_positional_encoding(
    seq_len: int,
    d_model: int,
) -> jax.Array:
    """Generate standard sinusoidal 1D positional encodings."""
    positions = jnp.arange(seq_len, dtype=jnp.float32)[:, None]
    dim_indices = jnp.arange(0, d_model, 2, dtype=jnp.float32)[None, :]
    frequencies = jnp.exp(-dim_indices * (jnp.log(10000.0) / d_model))
    angles = positions * frequencies

    pe = jnp.zeros((seq_len, d_model), dtype=jnp.float32)
    pe = pe.at[:, 0::2].set(jnp.sin(angles))
    pe = pe.at[:, 1::2].set(jnp.cos(angles))
    return pe


class TransformerBlock(nnx.Module):
    """Pre-LayerNorm Transformer Encoder Block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=d_model,
            decode=False,
            rngs=rngs,
        )
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.mlp_fc1 = nnx.Linear(d_model, mlp_dim, rngs=rngs)
        self.mlp_fc2 = nnx.Linear(mlp_dim, d_model, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Pre-LN Self-Attention
        norm_x = self.ln1(x)
        attn_out = self.attn(norm_x)
        x = x + attn_out

        # Pre-LN Feed-Forward MLP
        norm_x2 = self.ln2(x)
        mlp_out = self.mlp_fc2(nnx.gelu(self.mlp_fc1(norm_x2)))
        x = x + mlp_out
        return x


class TransformerStage1Estimator(nnx.Module):
    """Transformer + MLP Parameter Estimator for Stage 1.

    Maps received raw IQ bursts (B, L, 2) to normalized parameter vector (B, 11) in [0, 1].
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_params: int = 11,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_dim: int = 128,
        head_hidden_dim: int = 128,
        max_seq_len: int = 1024,
        use_learnable_pe: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.use_learnable_pe = use_learnable_pe
        self.max_seq_len = max_seq_len

        # 1. Project raw IQ features (I, Q) -> d_model tokens
        self.input_proj = nnx.Linear(in_channels, d_model, rngs=rngs)

        # 2. Positional Encodings
        if use_learnable_pe:
            self.pos_emb: nnx.Param[jax.Array] = nnx.Param(
                jax.random.normal(rngs.params(), (max_seq_len, d_model)) * 0.02
            )
        else:
            self.sin_pe = get_sinusoidal_positional_encoding(max_seq_len, d_model)

        # 3. Stacked Transformer Encoder Blocks
        self.blocks = nnx.List([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                rngs=rngs,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nnx.LayerNorm(d_model, rngs=rngs)

        # 4. Regression MLP Head: d_model -> head_hidden_dim -> out_params (11)
        self.head_fc1 = nnx.Linear(d_model, head_hidden_dim, rngs=rngs)
        self.head_fc2 = nnx.Linear(head_hidden_dim, out_params, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            x: Input burst of shape (B, L, 2) containing real and imaginary components.

        Returns:
            Normalized predicted parameters of shape (B, 11) in [0, 1].
        """
        seq_len = x.shape[1]

        # Project raw IQ channels: (B, L, 2) -> (B, L, d_model)
        tokens = self.input_proj(x)

        # Add positional encodings
        if self.use_learnable_pe:
            pe = self.pos_emb[:seq_len]
        else:
            pe = self.sin_pe[:seq_len]

        h = tokens + pe[None, :, :]

        # Pass through Transformer encoder blocks
        for block in self.blocks:
            h = block(h)

        h = self.final_norm(h)

        # Global average pooling across burst sequence length: (B, L, d_model) -> (B, d_model)
        h_pooled = jnp.mean(h, axis=1)

        # MLP Regression Head
        out = self.head_fc2(nnx.gelu(self.head_fc1(h_pooled)))

        # Sigmoid bounds parameters to normalized [0, 1] interval
        return nnx.sigmoid(out)
