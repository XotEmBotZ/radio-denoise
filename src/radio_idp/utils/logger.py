"""Experiment logging with TensorBoard and structured JSONL."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from tensorboardX import SummaryWriter


class ExperimentLogger:
    """Manages experiment logging to TensorBoard and structured JSONL logs."""

    def __init__(
        self,
        log_dir: Path | str = "runs",
        experiment_name: str = "experiment",
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(log_dir) / f"{experiment_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(logdir=str(self.run_dir))
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self._jsonl_file = open(self.jsonl_path, "a", encoding="utf-8")

    def log_metrics(
        self,
        step: int,
        metrics: dict[str, Any],
        prefix: str = "",
    ) -> None:
        """Log a dictionary of scalar metrics to TensorBoard and JSONL.

        Args:
            step: Training iteration step or epoch index.
            metrics: Dictionary of metric names to values.
            prefix: Optional category prefix (e.g. 'train', 'val').
        """
        now_iso = datetime.now().isoformat()
        json_record: dict[str, Any] = {
            "step": step,
            "timestamp": now_iso,
        }

        for key, val in metrics.items():
            if val is None:
                continue
            try:
                num_val = float(val)
            except (TypeError, ValueError):
                continue

            tag = f"{prefix}/{key}" if prefix else key
            self.writer.add_scalar(tag, num_val, step)
            json_record[tag] = num_val

        self._jsonl_file.write(json.dumps(json_record) + "\n")
        self._jsonl_file.flush()

    def flush(self) -> None:
        """Flush buffer contents to disk."""
        self.writer.flush()
        self._jsonl_file.flush()

    def close(self) -> None:
        """Close logger resources."""
        self.writer.flush()
        self.writer.close()
        if not self._jsonl_file.closed:
            self._jsonl_file.close()
