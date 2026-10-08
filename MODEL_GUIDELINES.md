# Model Iteration Guidelines

This repository follows a **Report-Persisted, Code-Tagged** workflow. We do **not** maintain legacy model code in `src/`. The active codebase only contains the latest architecture.

---

## 1. Directory Ownership

| Path | Contents | In Git? |
| :--- | :--- | :--- |
| `src/` | **Only** current active model code (freely refactor/overwrite). | Yes |
| `runs/<run_id>/` | Training curves (`metrics.jsonl`, event files). | Yes |
| `reports/<model_name>/` | Final PDF report, `benchmark_summary.json`, core plots. | Yes |
| `checkpoints/` | Model weights / state tensors. | **No** (`.gitignore`) |
| `data/` | Datasets. | **No** (`.gitignore`) |

---

## 2. Iteration Workflow (Lifecycle of a Model)

When finishing an architecture and before starting the next:

1. **Benchmark & Report**:
   - Run benchmark and save summary to `reports/<model_name>/`.
   - Place the final report PDF into `reports/<model_name>/`.
2. **Commit Code & Results**:
   ```bash
   git add src/ runs/<run_id>/ reports/<model_name>/
   git commit -m "feat(<model_name>): complete model, metrics, and report"
   ```
3. **Tag & Push Tag**:
   ```bash
   git tag -a <model-tag-name> -m "Model snapshot: <model_name>"
   git push origin <model-tag-name>
   ```
4. **Move to Next Model**:
   - Overwrite / refactor `src/` directly for the next architecture.
   - Do **not** keep dead code or backward-compatibility wrappers.

---

## 3. Restoring Old Models

To inspect, run, or reproduce any previous model:
```bash
git checkout <model-tag-name>
```
