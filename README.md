# Transformer Workbench

A Transformer workbench for profiling and optimizing language-model training and inference while preserving correctness and model quality.

## Setup

Recreate the local development environment:

```bash
uv sync
```

Authenticate the Modal client once per machine:

```bash
uv run modal setup
```

## Modal GPU smoke test

Launch the pinned GPU environment and print its software, hardware, seed, and Git
metadata as structured JSON:

```bash
uv run modal run scripts/modal_smoke.py
```
