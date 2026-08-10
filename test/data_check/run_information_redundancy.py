#!/usr/bin/env python3
"""Run information_redundancy check on generated scenario data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from test.data_check.information_redundancy import run_information_redundancy


def main() -> None:
    p = argparse.ArgumentParser(description="Information redundancy check (calibration split)")
    p.add_argument("--config", default="ieee5", help="Config stem under config/")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--no-generate-splits", action="store_true")
    args = p.parse_args()
    payload = run_information_redundancy(
        args.config,
        out_dir=args.out_dir,
        generate_splits=not args.no_generate_splits,
    )
    print(json.dumps({"passed": payload["passed"], "verdict": payload["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
