"""U-bank audit and continuous terminal-rule freeze."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.control.continuous_uctrl_amplitude import OUT, ROOT
from src.control.terminal_rule import FrozenTerminalRule, load_frozen_terminal_rule


U_BANK_AUDIT: dict[str, Any] = {
    "U_n_generation": (
        "For each support particle θ_n, U_n = u_req(θ_n) is the smallest value on "
        "the discrete control candidate grid that satisfies ROCOF and frequency-nadir "
        "constraints under the configured contingency (see src/control/banks.py "
        "generate_control_bank_for_split / u_req_for_theta)."
    ),
    "control_search_method": "sequential_scan_of_u_candidates_smallest_safe",
    "U_n_nature": "B_discrete_grid_selected",
    "continuous_u_ctrl_status": "approximation_based_on_discrete_U_bank",
    "physically_validated": False,
    "validation_note": (
        "U_n itself is a discrete-grid safe injection level. Continuous "
        "u_ctrl = Q_{1-α}(U|w) + margin interpolates between those banked discrete "
        "levels in posterior-quantile space. Intermediate continuous values are "
        "NOT individually re-simulated against ROCOF/nadir for this study. Safety "
        "thresholds (ROCOF, nadir) and calibrated margin are unchanged; only "
        "snap_up is removed from the terminal selector."
    ),
    "safety_constraints_unchanged": [
        "max ROCOF",
        "frequency nadir",
        "calibrated additive margin",
        "alpha = 0.05",
    ],
    "code_paths": {
        "banks": "src/control/banks.py",
        "u_req": "src/control/u_req.py",
        "cuda": "src/control/cuda_control.py",
        "posterior": "src/control/posterior_ctrl.py",
    },
}


def write_u_bank_audit(out_dir: Path | None = None) -> Path:
    out_dir = out_dir or (OUT / "summary")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "u_bank_audit.json"
    path.write_text(json.dumps(U_BANK_AUDIT, indent=2), encoding="utf-8")
    md = out_dir / "u_bank_audit.md"
    md.write_text(
        "\n".join(
            [
                "# U-bank and continuous u_ctrl validity audit",
                "",
                f"**U_n nature:** `{U_BANK_AUDIT['U_n_nature']}`",
                "",
                U_BANK_AUDIT["U_n_generation"],
                "",
                f"**Continuous u_ctrl status:** {U_BANK_AUDIT['continuous_u_ctrl_status']}",
                "",
                U_BANK_AUDIT["validation_note"],
                "",
                "Safety thresholds are not weakened. Margin/α reused from the frozen "
                "historical rule; only snap_up is disabled for this experiment version.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def freeze_continuous_terminal_rule(system: str) -> FrozenTerminalRule:
    """Copy historical α/margin/grid with snap_up=False; new hash."""
    exp = ROOT / "experiments" / f"{system}_T3"
    base = load_frozen_terminal_rule(exp)
    continuous = base.as_continuous(
        source=f"{base.source}|continuous_uctrl_amplitude_study"
    )
    cfg_dir = OUT / f"{system}_T3" / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "rule": {
            "alpha": continuous.alpha,
            "margin": continuous.margin,
            "quantile_level": continuous.quantile_level,
            "u_candidates": list(continuous.u_candidates),
            "snap_up": False,
            "formula": "Q_{1-alpha}(U|w) + margin",
            "parent_snapped_hash": base.terminal_rule_hash,
            "study": "continuous_uctrl_amplitude_adaptive_value",
        },
        **continuous.metadata(),
        "u_bank_audit_ref": "summary/u_bank_audit.json",
        "continuous_u_ctrl_is_physically_validated": False,
        "continuous_u_ctrl_is_approximation": True,
    }
    path = cfg_dir / "continuous_terminal_rule.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (cfg_dir / "terminal_rule_hash.txt").write_text(
        continuous.terminal_rule_hash + "\n", encoding="utf-8"
    )
    return continuous
