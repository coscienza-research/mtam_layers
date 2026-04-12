# Reproducibility Guide

This document defines a reproducible protocol for running quality checks on this repository.

## 1. Environment

- Python: 3.10, 3.11, or 3.12
- Install from repository root:

```bash
pip install -e .[dev]
```

Development dependencies include:
- pytest
- pytest-cov
- ruff

## 2. Deterministic Baseline

PowerShell:

```powershell
$env:PYTHONHASHSEED = "0"
```

Python seed setup:

```python
import torch

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
```

Notes:
- Deterministic algorithms can reduce performance.
- Some GPU operations may require additional backend-specific settings.

## 3. Quality Gates

Run lint:

```bash
ruff check .
```

Run tests with coverage threshold enforced (>=95%):

```bash
pytest
```

Coverage and threshold are configured in project metadata:
- pyproject coverage settings
- pytest addopts with cov-fail-under

## 4. CI Reproduction

CI reproduces the same checks on Linux for Python 3.10/3.11/3.12.
Local reproduction command sequence:

```bash
pip install -e .[dev]
ruff check .
pytest
```

## 5. Expected Artifacts

- Ruff: no violations
- Pytest: all tests passing
- Coverage: total >= 95%
