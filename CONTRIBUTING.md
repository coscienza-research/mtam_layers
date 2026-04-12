# Contributing

Thanks for your interest in improving mTAM Layers.

## Development Setup

1. Create and activate a Python 3.10+ environment.
2. Install development dependencies:

```bash
pip install -e .[dev]
```

## Quality Checklist

Before opening a pull request, run:

```bash
ruff check .
mypy
pytest
```

All three checks must pass.

## Pull Request Guidelines

- Keep commits atomic and focused.
- Include or update tests for behavioral changes.
- Update docs when changing public APIs.
- Prefer backward-compatible changes unless a breaking change is intentional and documented.

## Commit Message Style

Use concise imperative messages, for example:
- `Add entropy-gated diagnostics helper`
- `Fix shape validation for decoupled engine`

## Reporting Issues

Open an issue with:
- a minimal reproducible example,
- expected behavior,
- observed behavior,
- environment details (OS, Python, torch versions).
