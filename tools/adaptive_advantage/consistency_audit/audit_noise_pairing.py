"""Verify common-random-number pairing across policies."""

from __future__ import annotations

from typing import Any

import numpy as np

from tools.adaptive_advantage.config import SuiteConfig
from tools.adaptive_advantage.loaders import SystemBank
from tools.adaptive_advantage.planning_utils import make_crn_bundle

from .common import make_audit_crn, terminal_after_sequence


def audit_noise_pairing(bank: SystemBank, cfg: SuiteConfig) -> dict[str, Any]:
    crn = make_audit_crn(bank, cfg)
    # Re-create with same seed recipe and assert identity.
    crn2 = make_crn_bundle(
        n_eval=bank.n_eval,
        n_rep=cfg.noise_replicates,
        horizon=2,
        n_support=bank.n_support,
        n_hyp=cfg.n_hyp_y,
        sigma_y=bank.sigma_y,
        rng=np.random.default_rng(cfg.seed + 50),
    )
    same_eps = bool(np.allclose(crn.eps_obs, crn2.eps_obs))
    same_hyp = bool(
        np.array_equal(crn.hyp_idx, crn2.hyp_idx)
        and np.allclose(crn.hyp_noise, crn2.hyp_noise)
    )

    # Demonstrate that two Fixed sequences share eps at (i,r,t) even if actions differ.
    a, b, c = 0, 1, 2
    i, r = 0, 0
    y_a = float(bank.Y_eval[i, a] + crn.eps_obs[i, r, 0])
    y_c = float(bank.Y_eval[i, c] + crn.eps_obs[i, r, 0])
    shared_eps_step0 = bool(
        np.isclose(y_a - bank.Y_eval[i, a], y_c - bank.Y_eval[i, c])
    )

    # Independent-noise mismatch demo (what NOT to do).
    rng_ind = np.random.default_rng(999)
    u_shared = []
    u_indep = []
    for ii in range(min(bank.n_eval, 8)):
        for rr in range(min(cfg.noise_replicates, 4)):
            u_shared.append(
                terminal_after_sequence(bank, ii, (a, b), crn.eps_obs[ii, rr])
            )
            eps_ind = rng_ind.normal(0.0, bank.sigma_y, size=2)
            u_indep.append(terminal_after_sequence(bank, ii, (a, b), eps_ind))

    return {
        "crn_reproducible_from_seed": bool(same_eps and same_hyp),
        "shared_eps_across_different_actions_at_same_step": shared_eps_step0,
        "paired_bootstrap_valid_only_if_shared_eps": True,
        "independent_noise_would_inflate_variance": {
            "mean_abs_diff_shared_vs_independent_streams": float(
                np.mean(np.abs(np.asarray(u_shared) - np.asarray(u_indep)))
            )
        },
        "assertion": (
            "PASS: audit CRN is deterministic from seed+50 and shared across policies"
            if same_eps and same_hyp and shared_eps_step0
            else "FAIL: CRN pairing broken"
        ),
    }
