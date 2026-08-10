"""CUDA batch simulator consistency checks (requires PyCUDA + GPU).

Enable with: RUN_CUDA_BATCH_TEST=1 pytest test/test_cuda_batch.py
Or: python -m test.test_cuda_batch
"""

from __future__ import annotations

import os

import pytest


def test_pycuda_matches_cpu_smoke():
    if os.environ.get("RUN_CUDA_BATCH_TEST") != "1":
        pytest.skip("Set RUN_CUDA_BATCH_TEST=1 to run GPU/PyCUDA check")
    from src.swing_equation_ode.cuda_batch import profile_batch_sizes, test_pycuda_matches_cpu

    stats = test_pycuda_matches_cpu()
    assert isinstance(stats, dict)
    profile_batch_sizes()


def run() -> None:
    from src.swing_equation_ode.cuda_batch import profile_batch_sizes, test_pycuda_matches_cpu

    stats = test_pycuda_matches_cpu()
    print(stats)
    profile_batch_sizes()


if __name__ == "__main__":
    run()
