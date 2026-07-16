"""Information redundancy check orchestration (tests/data_check)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.config import SBOEDConfig, load_config_for_run, repo_root
from src.data import resolve_data_path, save_json
from src.swing_equation_ode.design import build_catalog
from src.table_scoring import TableThetaSupport
from tests.data_check.bootstrap import bootstrap_mean_ci_threshold
from tests.data_check.lookahead_oracle import run_certification_protocol_t2
from tests.data_check.oracle_eig import greedy_mc_eig_action
from tests.data_check.redundancy import build_action_centres, compute_redundancy_diagnostics, save_redundancy_artifacts
from tests.data_check.splits import (
    CERTIFICATION_SPLIT,
    SEARCH_SPLIT,
    SUPPORT_SPLIT,
    ensure_redundancy_splits,
    load_redundancy_split_systems,
    redundancy_split_config,
)


@dataclass(frozen=True)
class InformationRedundancyConfig:
    horizon: int = 2
    particle_count: int = 256
    predictive_mc_samples: int = 128
    noise_replicas: int = 8
    bootstrap_replicates: int = 1000
    min_greedy_gap_nats: float | None = None
    min_adaptivity_gap_nats: float | None = None
    confidence_level: float = 0.95
    certification_seeds: tuple[int, ...] = (0, 1, 2)
    action_subset: list[int] | None = None
    require_adaptivity: bool = True
    support_split: str = SUPPORT_SPLIT
    search_split: str = SEARCH_SPLIT
    certification_split: str = CERTIFICATION_SPLIT

    @classmethod
    def from_cfg(cls, cfg: SBOEDConfig) -> InformationRedundancyConfig:
        g = dict(cfg.raw.get("information_redundancy") or cfg.raw.get("gate1") or {})
        split_roles = redundancy_split_config(cfg)["roles"]
        particle_count = int(g.get("particle_count", cfg.prior.get("mc_samples", 256)))
        base = cls(particle_count=particle_count)
        if "horizon" in g:
            base = dataclass_replace(base, horizon=int(g["horizon"]))
        if "predictive_mc_samples" in g:
            base = dataclass_replace(base, predictive_mc_samples=int(g["predictive_mc_samples"]))
        if "noise_replicas" in g:
            base = dataclass_replace(base, noise_replicas=int(g["noise_replicas"]))
        if "bootstrap_replicates" in g:
            base = dataclass_replace(base, bootstrap_replicates=int(g["bootstrap_replicates"]))
        if "confidence_level" in g:
            base = dataclass_replace(base, confidence_level=float(g["confidence_level"]))
        seeds_key = "certification_seeds" if "certification_seeds" in g else "calibration_seeds"
        if seeds_key in g:
            base = dataclass_replace(base, certification_seeds=tuple(int(x) for x in g[seeds_key]))
        if "action_subset" in g:
            base = dataclass_replace(base, action_subset=list(g["action_subset"]))
        if "require_adaptivity" in g:
            base = dataclass_replace(base, require_adaptivity=bool(g["require_adaptivity"]))
        base = dataclass_replace(
            base,
            support_split=str(g.get("support_split", split_roles["support"])),
            search_split=str(g.get("search_split", split_roles["search"])),
            certification_split=str(g.get("certification_split", split_roles["certification"])),
        )
        log_n = math.log(max(particle_count, 2))
        min_g = float(g.get("min_greedy_gap_nats", 0.05 * log_n))
        min_a = float(g.get("min_adaptivity_gap_nats", 0.02 * log_n))
        return dataclass_replace(base, min_greedy_gap_nats=min_g, min_adaptivity_gap_nats=min_a)


def dataclass_replace(obj, **kwargs):
    from dataclasses import replace
    return replace(obj, **kwargs)


def run_information_redundancy(
    config_name: str,
    *,
    out_dir: Path | None = None,
    project_root: Path | None = None,
    generate_splits: bool = True,
) -> dict[str, Any]:
    root = project_root or repo_root()
    cfg = load_config_for_run(config_name, root, step_number=2)
    ir_cfg = InformationRedundancyConfig.from_cfg(cfg)
    data_path = resolve_data_path(root, cfg)
    if generate_splits:
        ensure_redundancy_splits(root, cfg)

    support_pool = load_redundancy_split_systems(data_path, ir_cfg.support_split)
    search_systems = load_redundancy_split_systems(data_path, ir_cfg.search_split)
    certification = load_redundancy_split_systems(data_path, ir_cfg.certification_split)
    catalog = build_catalog(cfg)
    n_actions = len(catalog)
    sigma_y = cfg.sigma_y

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = out_dir or (
        root / "tests" / "data_check" / "results" / f"{stamp}_{cfg.run_slug}_information_redundancy"
    )
    out.mkdir(parents=True, exist_ok=True)

    seed_results: list[dict[str, Any]] = []
    for seed in ir_cfg.certification_seeds:
        rng = np.random.default_rng(seed)
        support_rng = np.random.default_rng(int(cfg.prior.get("mc_support_seed", seed)))
        table_support = TableThetaSupport.from_train(
            support_pool,
            cfg,
            support_rng,
            n_particles=ir_cfg.particle_count,
        )
        centres = build_action_centres(table_support, n_actions)
        mc_greedy, _eig_results = greedy_mc_eig_action(
            table_support.log_p0,
            centres,
            sigma_y,
            ir_cfg.predictive_mc_samples,
            rng,
            feasible=np.asarray(ir_cfg.action_subset) if ir_cfg.action_subset else None,
        )
        plug_myopic, _ = _plug_in_first_action(table_support, sigma_y, n_actions)

        redundancy = compute_redundancy_diagnostics(
            table_support,
            n_actions,
            sigma_y,
            ir_cfg.predictive_mc_samples,
            rng,
            action_subset=ir_cfg.action_subset,
        )
        save_redundancy_artifacts(out / f"seed_{seed}", redundancy, {"seed": seed})

        cert = run_certification_protocol_t2(
            table_support.log_p0,
            table_support,
            certification,
            sigma_y,
            ir_cfg.predictive_mc_samples,
            rng,
            n_actions,
            noise_replicas=ir_cfg.noise_replicas,
            action_subset=ir_cfg.action_subset,
        )

        g_full_ci, g_full_pass = bootstrap_mean_ci_threshold(
            cert.per_system_g_full,
            ir_cfg.min_greedy_gap_nats,
            n_replicates=ir_cfg.bootstrap_replicates,
            confidence_level=ir_cfg.confidence_level,
            rng=np.random.default_rng(seed + 1000),
        )
        g_adapt_ci, g_adapt_pass = bootstrap_mean_ci_threshold(
            cert.per_system_g_adapt,
            ir_cfg.min_adaptivity_gap_nats,
            n_replicates=ir_cfg.bootstrap_replicates,
            confidence_level=ir_cfg.confidence_level,
            rng=np.random.default_rng(seed + 2000),
        )

        seed_results.append({
            "seed": seed,
            "mc_greedy_first_action": int(mc_greedy),
            "plug_myopic_first_action": int(plug_myopic),
            "oracle_first_action": cert.oracle_first_action,
            "first_action_mismatch": cert.oracle_first_action != plug_myopic,
            "v_star_planning": cert.v_star_planning,
            "v_oracle": cert.v_oracle,
            "v_myopic": cert.v_myopic,
            "v_fixed": cert.v_fixed,
            "g_full": cert.g_full,
            "g_adapt": cert.g_adapt,
            "g_full_ci": g_full_ci.__dict__,
            "g_adapt_ci": g_adapt_ci.__dict__,
            "g_full_pass": g_full_pass,
            "g_adapt_pass": g_adapt_pass,
            "noise_replicas": cert.noise_replicas,
            "n_certification_systems": cert.n_certification_systems,
            "best_fixed_sequence": list(cert.best_fixed_sequence),
            "top_eig_actions": redundancy.top_eig_actions[:10],
            "branch_report": cert.branch_report,
            "split_roles": {
                "support": ir_cfg.support_split,
                "search": ir_cfg.search_split,
                "certification": ir_cfg.certification_split,
                "n_support_pool": len(support_pool),
                "n_search": len(search_systems),
            },
        })

    all_g_full = all(r["g_full_pass"] for r in seed_results)
    all_g_adapt = all(r["g_adapt_pass"] for r in seed_results) if ir_cfg.require_adaptivity else True
    non_myopic = all_g_full
    adaptive_certified = all_g_adapt
    passed = non_myopic and adaptive_certified

    if passed:
        verdict = (
            "PASS: certification split shows positive paired realized G_full and G_adapt "
            "with virtual noise replicas."
        )
    elif non_myopic and not adaptive_certified:
        verdict = (
            "PARTIAL: scenario supports a non-myopic planning advantage (G_full > 0 on certification), "
            "but history-adaptive advantage (G_adapt > 0) is not certified. "
            "Do not claim adaptive-DAD superiority yet."
        )
    else:
        verdict = (
            "FAIL: The reset-based scenario may remain physically valid, but it is not a "
            "valid DAD-superiority benchmark under the current prior, action catalogue, "
            "noise level, and horizon."
        )

    payload: dict[str, Any] = {
        "check": "information_redundancy",
        "passed": passed,
        "non_myopic_certified": non_myopic,
        "adaptive_certified": adaptive_certified,
        "config_name": config_name,
        "run_slug": cfg.run_slug,
        "data_path": str(data_path),
        "information_redundancy_config": ir_cfg.__dict__,
        "seed_results": seed_results,
        "verdict": verdict,
    }
    save_json(payload, out / "information_redundancy_results.json")
    _write_information_redundancy_report(out / "information_redundancy_report.md", payload, cfg)
    return payload


def _plug_in_first_action(
    table_support: TableThetaSupport,
    sigma_y: float,
    n_actions: int,
) -> tuple[int, dict[int, float]]:
    from src.contrastive.spce import clamp_info_gain, log_gaussian_observation_density, normalize_log_weights, posterior_entropy
    from src.table_scoring import y_sim_last_step_from_tables

    log_unnorm = np.array(table_support.log_p0, dtype=np.float64)
    p_before = normalize_log_weights(log_unnorm)
    h_before = posterior_entropy(p_before)
    best_a, best_dh = 0, -np.inf
    scores: dict[int, float] = {}
    for a in range(n_actions):
        m_vals = y_sim_last_step_from_tables(table_support, [a])
        y_hat = float(np.sum(p_before * m_vals))
        log_L = log_gaussian_observation_density(y_hat, m_vals, sigma_y)
        p_after = normalize_log_weights(log_unnorm + log_L)
        dh = clamp_info_gain(h_before - posterior_entropy(p_after))
        scores[a] = dh
        if dh > best_dh:
            best_dh, best_a = dh, a
    return int(best_a), scores


def _write_information_redundancy_report(path: Path, payload: dict[str, Any], cfg: SBOEDConfig) -> None:
    lines = [
        f"# Information redundancy report — {cfg.run_slug}",
        "",
        f"**Result:** {'PASS' if payload['passed'] else 'PARTIAL/FAIL'}",
        "",
        payload["verdict"],
        "",
        f"- Non-myopic certified (G_full): {payload['non_myopic_certified']}",
        f"- Adaptive certified (G_adapt): {payload['adaptive_certified']}",
        "",
        "## Per-seed summary",
    ]
    for r in payload["seed_results"]:
        lines.extend([
            f"### Seed {r['seed']}",
            f"- V_oracle = {r['v_oracle']:.4f}, V_myopic = {r['v_myopic']:.4f}, G_full = {r['g_full']:.4f}",
            f"- V_fixed = {r['v_fixed']:.4f}, G_adapt = {r['g_adapt']:.4f}",
            f"- V*_planning (Q2 diagnostic) = {r['v_star_planning']:.4f}",
            f"- Oracle first = {r['oracle_first_action']}, myopic first = {r['plug_myopic_first_action']}",
            f"- Noise replicas R = {r['noise_replicas']}, certification systems M = {r['n_certification_systems']}",
            f"- G_full CI [{r['g_full_ci']['lower']:.4f}, {r['g_full_ci']['upper']:.4f}] pass={r['g_full_pass']}",
            f"- G_adapt CI [{r['g_adapt_ci']['lower']:.4f}, {r['g_adapt_ci']['upper']:.4f}] pass={r['g_adapt_pass']}",
            "",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
