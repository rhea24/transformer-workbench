"""Verify the Modal GPU environment and print reproducibility metadata."""

from __future__ import annotations

import json
import platform
import subprocess
from typing import Any

import modal


APP_NAME = "transformer-workbench-smoke"
PYTHON_VERSION = "3.11"
PYTORCH_VERSION = "2.11.0"
NUMPY_VERSION = "2.3.3"
GPU_TYPE = "L4"
DEFAULT_SEED = 12

image = modal.Image.debian_slim(python_version=PYTHON_VERSION).uv_pip_install(
    f"torch=={PYTORCH_VERSION}",
    f"numpy=={NUMPY_VERSION}",
)
app = modal.App(APP_NAME, image=image)


def _run_local_command(args: list[str]) -> str | None:
    """Return command output, or ``None`` when metadata is unavailable."""
    try:
        result = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_metadata() -> dict[str, Any]:
    status = _run_local_command(["git", "status", "--porcelain"])
    return {
        "commit": _run_local_command(["git", "rev-parse", "HEAD"]) or "unknown",
        "dirty": None if status is None else bool(status),
    }


@app.function(gpu=GPU_TYPE, timeout=300)
def collect_environment(
    seed: int,
    git_commit: str,
    git_dirty: bool | None,
) -> str:
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Modal allocated a function without a CUDA-capable GPU")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    driver_version = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    report = {
        "software": {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "pytorch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "nvidia_driver": driver_version,
            "numpy": np.__version__,
        },
        "hardware": {
            "requested_gpu": GPU_TYPE,
            "detected_gpu": properties.name,
            "device_count": torch.cuda.device_count(),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
        },
        "run": {
            "seed": seed,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        },
    }
    return json.dumps(report, indent=2, sort_keys=True)


@app.local_entrypoint()
def main(seed: int = DEFAULT_SEED) -> None:
    git = _git_metadata()
    report_json = collect_environment.remote(
        seed=seed,
        git_commit=git["commit"],
        git_dirty=git["dirty"],
    )
    print(report_json)
