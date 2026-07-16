from __future__ import annotations

from src.swing_equation_ode.cuda_batch import profile_batch_sizes, test_pycuda_matches_cpu


def run() -> None:
    stats = test_pycuda_matches_cpu()
    print(stats)
    profile_batch_sizes()


if __name__ == "__main__":
    run()
