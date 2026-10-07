"""Orbax checkpoint manager wrapper for Flax NNX and JAX models."""

from pathlib import Path
from typing import Any

import flax.nnx as nnx
import orbax.checkpoint as ocp


class ModelCheckpointManager:
    """Manages model and optimizer checkpoints using Orbax."""

    def __init__(
        self,
        directory: Path | str,
        max_to_keep: int = 3,
    ):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=True,
        )
        self.mngr = ocp.CheckpointManager(
            self.directory,
            options=self.options,
        )

    def save(
        self,
        step: int,
        model: Any,
        opt_state: Any = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Save a checkpoint at the given training step.

        Args:
            step: Training step index.
            model: Flax nnx.Module or PyTree state dictionary.
            opt_state: Optional optimizer state PyTree.
            extra_metadata: Optional dictionary of scalar metrics or config data.

        Returns:
            True if checkpoint save was triggered.
        """
        if isinstance(model, nnx.Module):
            _, state = nnx.split(model)
            model_dict = state.to_pure_dict()
        elif isinstance(model, nnx.State):
            model_dict = model.to_pure_dict()
        else:
            model_dict = model

        save_dict = {
            "model": model_dict,
            "opt_state": opt_state,
            "extra_metadata": extra_metadata or {},
        }

        success = self.mngr.save(
            step,
            args=ocp.args.StandardSave(save_dict),
        )
        return bool(success)

    def restore_latest(
        self,
        model: nnx.Module | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Restore the latest available checkpoint.

        Args:
            model: Optional live nnx.Module to update in-place with restored weights.

        Returns:
            tuple of (step, restored_dict) where restored_dict contains 'model', 'opt_state', etc.
        """
        latest_step = self.mngr.latest_step()
        if latest_step is None:
            raise FileNotFoundError(f"No checkpoints found in {self.directory}")
        return self.restore_step(latest_step, model=model)

    def restore_step(
        self,
        step: int,
        model: nnx.Module | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Restore a specific checkpoint step.

        Args:
            step: Step index to restore.
            model: Optional live nnx.Module to update in-place.

        Returns:
            tuple of (step, restored_dict).
        """
        restored = self.mngr.restore(
            step,
            args=ocp.args.StandardRestore(),
        )

        if model is not None and isinstance(model, nnx.Module):
            nnx.update(model, restored["model"])

        return step, restored

    def latest_step(self) -> int | None:
        """Return the latest checkpoint step index, or None if no checkpoints exist."""
        return self.mngr.latest_step()

    def wait_until_finished(self) -> None:
        """Wait for asynchronous checkpoint write operations to complete."""
        self.mngr.wait_until_finished()
