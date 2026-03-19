# Local Python Environment

This repo uses a local virtual environment at `.venv/` for model training and evaluation.

## Why

- Keeps PyTorch, Ultralytics, and related packages isolated from the system Python
- Makes future LLM sessions use the same interpreter and package set
- Avoids mixing training dependencies with unrelated global packages

## Use It

Run tools from the repo root with the executables inside `.venv`:

```bash
.venv/bin/python --version
.venv/bin/pip list
.venv/bin/python scripts/compare_yolo_models.py --help
```

If you want an interactive shell session to use the venv first:

```bash
source .venv/bin/activate
python --version
```

## Installed Core Packages

- `torch`
- `torchvision`
- `torchaudio`
- `ultralytics`

## Notes For Future Sessions

- Prefer `.venv/bin/python` and `.venv/bin/pip` in commands for reproducibility
- Store training outputs under `workspace/` so the read-only AIS source is never touched
- If dependencies drift later, update this file with the new setup steps
