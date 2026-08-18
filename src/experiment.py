"""Config-driven experiment steps (called by scripts/*.sh).

  python -m src.experiment allocate-dir --config configs/ieee9.yaml
  python -m src.experiment generate-data --config configs/ieee9.yaml --exp-dir ...
  python -m src.experiment train --config configs/ieee9.yaml --method dad --exp-dir ...
  python -m src.experiment evaluate --config configs/ieee9.yaml --exp-dir ...

Result folders are always named:
  date_time_configname_Uctrl|EIG_Tnum_NobsN_sigmaX
  e.g. 07232026_215655_ieee14_Uctrl_T3_Nobs0_sigma0p005

Config YAML controls generation / compression / training / evaluation parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from src.config import (
    DEFAULT_STEP_NUMBER,
    effective_step_number,
    load_config,
    repo_root,
    resolve_config_path,
    with_step_number,
)
from src.banks.quality import BankQualityError
from src.banks.power_grid import (
    generate_physical_bank,
    resolve_dataset_dir,
)
from src.objectives.mocu.context import (
    ALL_METHOD_KEYS,
    EXTENDED_METHOD_KEYS,
    build_context_from_config,
    context_report_meta,
    method_display_name,
    methods_from_args,
    normalize_method_key,
)
from src.results.summary import write_objective_summary_md
from src.layout import (
    allocate_result_dir,
    ensure_result_layout,
    resolve_result_dir,
    write_run_config,
)

ExperimentType = Literal["objective_based", "eig_based"]
EXPERIMENT_TYPES: tuple[str, ...] = ("objective_based", "eig_based")


def load_experiment_config(
    config_arg: str,
    *,
    step_number: int | None = None,
    n_obs: int = 0,
    noise_sigma: float = 0.005,
):
    """Load YAML and apply CLI observation / horizon overrides."""
    root = repo_root()
    path = Path(config_arg)
    if path.suffix in (".yaml", ".yml") and path.is_file():
        cfg = load_config(path.resolve())
    else:
        cfg = load_config(resolve_config_path(config_arg, root))
    T = effective_step_number(step_number, default=DEFAULT_STEP_NUMBER)
    cfg = with_step_number(cfg, T)
    if int(n_obs) < 0:
        raise SystemExit(f"Invalid --N_obs: {n_obs} (non-negative integer required)")
    if float(noise_sigma) <= 0.0:
        raise SystemExit(
            f"Invalid --noise_sigma: {noise_sigma} (positive float required)"
        )
    obs = dict(cfg.raw.get("observation") or {})
    obs["N_obs"] = int(n_obs)
    obs["noise_sigma"] = float(noise_sigma)
    cfg.raw["observation"] = obs
    return cfg


def resolve_experiment_type(raw: str | None) -> ExperimentType:
    t = (raw or "objective_based").strip().lower().replace("-", "_")
    if t not in EXPERIMENT_TYPES:
        raise SystemExit(
            f"Invalid --experiment-type {raw!r} "
            f"(allowed: {', '.join(EXPERIMENT_TYPES)})"
        )
    return t  # type: ignore[return-value]


def _use_vector_eig_pipeline(cfg, exp_type: str, n_obs: int) -> bool:
    """Physical-bank vector EIG (incl. continuous max-ROCOF with N_obs=0).

    Table-lookup EIG (``pipeline.run_evaluation``) is used only for classic
    eig_based + N_obs=0 non-continuous IEEE configs.
    """
    if str(exp_type).lower().replace("-", "_") != "eig_based":
        return False
    if int(n_obs) > 0:
        return True
    return bool(getattr(cfg, "continuous_duration_mode", False))


def _add_experiment_type(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--experiment-type",
        "--experiment_type",
        dest="experiment_type",
        default="objective_based",
        choices=EXPERIMENT_TYPES,
        help="objective_based (default u_ctrl) or eig_based (terminal EIG)",
    )


def _add_exp_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--exp-dir",
        default=None,
        help=(
            "Result folder under experiments/ "
            "(name: date_time_configname_Uctrl|EIG_Tnum_NobsN_sigmaX). "
            "If omitted: allocate (generate/allocate-dir) or use latest match."
        ),
    )


def _add_seed(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seed",
        type=int,
        default=101,
        help="Training RNG seed recorded on this experiment (default: 101)",
    )


def _add_T(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-T",
        "--T",
        "--step-number",
        dest="step_number",
        type=int,
        default=None,
        metavar="T",
        help=f"Probe horizon (default {DEFAULT_STEP_NUMBER} if omitted)",
    )


def _add_observation_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--N_obs",
        "--n-obs",
        "--n_obs",
        dest="n_obs",
        type=int,
        default=0,
        metavar="N",
        help="Method-visible trajectory samples; 0 = scalar max-ROCOF (default: 0)",
    )
    parser.add_argument(
        "--noise_sigma",
        "--noise-sigma",
        dest="noise_sigma",
        type=float,
        default=0.005,
        metavar="SIGMA",
        help=(
            "Observation-noise standard deviation (default: 0.005; "
            "Hz/s when N_obs=0, Hz when N_obs>0)"
        ),
    )


def _resolve_exp_dir(
    cfg,
    exp_type: str,
    exp_dir_arg: str | None,
    *,
    create_new: bool,
) -> Path:
    return resolve_result_dir(
        cfg,
        exp_type,
        exp_dir=exp_dir_arg,
        create_new=create_new,
    )


def _run_method_keys(cfg, method_arg: str | None) -> list[str]:
    """Canonical methods for this experiment (CLI subset, else yaml)."""
    return methods_from_args(cfg, method_arg)


def _run_record_extra(
    args: argparse.Namespace,
    *,
    methods: list[str] | None = None,
    **more: Any,
) -> dict[str, Any]:
    extra = dict(more)
    if methods is not None:
        extra["methods"] = list(methods)
    seed = extra.get("seed", getattr(args, "seed", None))
    if seed is not None:
        extra["seed"] = int(seed)
    return extra


VECTOR_EIG_METHOD_MAP = {
    "dad": "dad_eig",
    "rl_sboed": "rl_sboed_eig",
    "moe_sboed": "moe_sboed",
    "matched_dense": "matched_dense",
    "myopic": "myopic_delta_h",
    "fixed": "fixed_open_loop",
    "random": "random",
}


def _evaluate_run_identity(
    args: argparse.Namespace,
) -> tuple[Any, str, Path, dict[str, Any], int]:
    """Load eval config from --exp-dir run_config when present."""
    from src.layout import (
        load_run_config_doc,
        method_checkpoint_available,
        resolve_eval_seed,
        run_methods_from_doc,
    )

    exp_type = resolve_experiment_type(args.experiment_type)
    step = args.step_number
    n_obs = args.n_obs
    sigma = args.noise_sigma
    run_doc: dict[str, Any] = {}
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
        if not exp_dir.is_absolute():
            exp_dir = repo_root() / exp_dir
        run_doc = load_run_config_doc(exp_dir)
        if run_doc.get("T") is not None:
            step = int(run_doc["T"])
        elif run_doc.get("step_number") is not None:
            step = int(run_doc["step_number"])
        if run_doc.get("N_obs") is not None:
            n_obs = int(run_doc["N_obs"])
        if run_doc.get("noise_sigma") is not None:
            sigma = float(run_doc["noise_sigma"])
        if run_doc.get("experiment_type"):
            exp_type = resolve_experiment_type(str(run_doc["experiment_type"]))
    cfg = load_experiment_config(
        args.config,
        step_number=step,
        n_obs=n_obs,
        noise_sigma=sigma,
    )
    exp_dir = _resolve_exp_dir(cfg, exp_type, args.exp_dir, create_new=False)
    run_doc = load_run_config_doc(exp_dir) or run_doc
    eval_seed = resolve_eval_seed(exp_dir, getattr(args, "seed", None))
    if args.method:
        method_keys = methods_from_args(cfg, args.method)
    else:
        stamped = run_methods_from_doc(run_doc)
        method_keys = stamped or methods_from_args(cfg, None)
    kept: list[str] = []
    for key in method_keys:
        if method_checkpoint_available(exp_dir, key):
            kept.append(key)
        else:
            print(f"[evaluate] skip {key}: missing .pth under {exp_dir / 'model'}")
    if not kept:
        raise SystemExit(f"No evaluable methods remain in {exp_dir}")
    return cfg, exp_type, exp_dir, {"methods": kept, "eval_seed": eval_seed, "run_doc": run_doc}, eval_seed


def cmd_allocate_dir(args: argparse.Namespace) -> None:
    """Print a newly allocated result folder path (for run.sh capture)."""
    cfg = load_experiment_config(
        args.config,
        step_number=args.step_number,
        n_obs=args.n_obs,
        noise_sigma=args.noise_sigma,
    )
    exp_type = resolve_experiment_type(args.experiment_type)
    if args.exp_dir:
        path = _resolve_exp_dir(cfg, exp_type, args.exp_dir, create_new=False)
    else:
        path = allocate_result_dir(cfg, exp_type)
    ensure_result_layout(path)
    # stdout is only the path so shells can capture it cleanly
    print(str(path))


def cmd_generate_data(args: argparse.Namespace) -> None:
    cfg = load_experiment_config(
        args.config,
        step_number=args.step_number,
        n_obs=args.n_obs,
        noise_sigma=args.noise_sigma,
    )
    exp_type = resolve_experiment_type(args.experiment_type)
    exp_dir = _resolve_exp_dir(
        cfg, exp_type, args.exp_dir, create_new=args.exp_dir is None
    )
    force = bool(getattr(args, "force", False))
    run_methods = _run_method_keys(cfg, getattr(args, "method", None))
    print(
        f"[generate-data] type={exp_type} config={cfg.config_path} "
        f"exp_dir={exp_dir} smoke={args.smoke} force={force} methods={run_methods}"
    )
    from src.domains.sir.context import is_sir_config

    if is_sir_config(cfg):
        if exp_type != "eig_based":
            raise SystemExit(
                "SIR ODE supports --experiment-type eig_based only "
                "(no MOCU/control track yet)."
            )
        from src.domains.sir.banks import generate_sir_bank, sir_bank_is_complete

        data_dir = resolve_dataset_dir(cfg)
        print(f"[generate-data] SIR ODE dataset_dir={data_dir}")
        if force:
            raise SystemExit(
                "SIR experiments are databank-only: --force regeneration is "
                "disabled. Reuse the existing data/sir_ode bank."
            )
        if not sir_bank_is_complete(data_dir):
            raise SystemExit(
                f"SIR databank missing or incomplete at {data_dir}. "
                "The experiment pipeline will not simulate trajectories on the fly."
            )
        rep = generate_sir_bank(
            cfg, smoke=args.smoke, force=False
        )
        write_run_config(
            exp_dir,
            cfg,
            data_dir,
            experiment_type=exp_type,
            extra=_run_record_extra(
                args,
                methods=run_methods,
                observation_model="sir_infected_count_gaussian",
                domain="sir_ode",
                design="measurement_time_chronological_on_idad_grid",
                N_obs=1,
                data_generation={
                    **dict(cfg.raw.get("data_generation") or {}),
                    "reused": bool(rep.get("reused")),
                    "force": force,
                    "smoke": bool(args.smoke),
                    "bank_shape_train": rep.get("bank_shape_train"),
                    "bank_shape_test": rep.get("bank_shape_test"),
                    "elapsed_seconds": rep.get("elapsed_seconds"),
                    "train_theta_count": rep.get("train_theta_count"),
                    "test_theta_count": rep.get("test_theta_count"),
                },
            ),
        )
        print(
            f"[generate-data] reused={bool(rep.get('reused'))} "
            f"path={rep.get('path')} "
            f"train={rep.get('train_theta_count')} test={rep.get('test_theta_count')} "
            f"actions={rep.get('n_actions')}"
        )
        print(f"EXP_DIR={exp_dir}")
        return

    if exp_type == "objective_based" or _use_vector_eig_pipeline(
        cfg, exp_type, int(args.n_obs)
    ):
        from src.banks.power_grid import bank_has_max_rocof, bank_is_complete

        data_dir = resolve_dataset_dir(cfg)
        print(f"[generate-data] dataset_dir={data_dir}")
        if force:
            raise SystemExit(
                "Experiments are databank-only: --force regeneration is disabled. "
                "Reuse the existing physical bank."
            )
        if not bank_is_complete(data_dir):
            raise SystemExit(
                f"Physical databank missing or incomplete at {data_dir}. "
                "The experiment pipeline will not simulate trajectories on the fly."
            )
        if not bank_has_max_rocof(data_dir):
            raise SystemExit(
                f"Physical databank at {data_dir} is missing max_rocof arrays "
                "required by the shared bank contract. Complete the bank "
                "offline first."
            )
        rep = generate_physical_bank(cfg, smoke=args.smoke, force=False)
        if exp_type == "eig_based" and int(args.n_obs) == 0:
            obs_model = "continuous_duration_max_rocof"
        elif exp_type == "eig_based":
            obs_model = "sampled_delta_f_vector"
        else:
            obs_model = "objective_delta_f_vector"
        write_run_config(
            exp_dir,
            cfg,
            data_dir,
            experiment_type=exp_type,
            extra=_run_record_extra(
                args,
                methods=run_methods,
                observation_model=obs_model,
                N_obs=int(args.n_obs),
                data_generation={
                    **dict(cfg.raw.get("data_generation") or {}),
                    "reused": bool(rep.get("reused")),
                    "force": force,
                    "smoke": bool(args.smoke),
                    "bank_shape_train": rep.get("bank_shape_train")
                    or rep.get("observation_shape"),
                    "bank_shape_test": rep.get("bank_shape_test"),
                    "elapsed_seconds": rep.get("elapsed_seconds"),
                    "N_sim": rep.get("N_sim"),
                    "train_theta_count": rep.get("train_theta_count"),
                    "test_theta_count": rep.get("test_theta_count"),
                },
            ),
        )
        print(
            f"[generate-data] reused={bool(rep.get('reused'))} "
            f"type={exp_type} train={rep.get('train_theta_count')} "
            f"test={rep.get('test_theta_count')} "
            f"actions={rep.get('n_actions')}"
        )
        print(f"EXP_DIR={exp_dir}")
        return

    from src.objectives.eig.pipeline import generate_tables, print_experiment_banner

    root = repo_root()
    linked, data_path, train_systems, test_systems = generate_tables(
        cfg, root, exp_dir, experiment_type=exp_type
    )
    print_experiment_banner(
        cfg, linked, data_path, train_systems, test_systems, cfg.methods
    )
    print(
        f"[generate-data] type={exp_type} "
        f"train={len(train_systems)} test={len(test_systems)} "
        f"data={data_path}"
    )
    print(f"EXP_DIR={linked}")


def cmd_train(args: argparse.Namespace) -> None:
    from src.layout import load_run_config_doc, resolve_eval_seed

    exp_type = resolve_experiment_type(args.experiment_type)
    step = args.step_number
    n_obs = args.n_obs
    sigma = args.noise_sigma
    train_seed = int(args.seed)
    if args.exp_dir:
        exp_guess = Path(args.exp_dir)
        if not exp_guess.is_absolute():
            exp_guess = repo_root() / exp_guess
        run_doc = load_run_config_doc(exp_guess)
        if run_doc.get("T") is not None:
            step = int(run_doc["T"])
        elif run_doc.get("step_number") is not None:
            step = int(run_doc["step_number"])
        if run_doc.get("N_obs") is not None:
            n_obs = int(run_doc["N_obs"])
        if run_doc.get("noise_sigma") is not None:
            sigma = float(run_doc["noise_sigma"])
        if run_doc.get("experiment_type"):
            exp_type = resolve_experiment_type(str(run_doc["experiment_type"]))
        train_seed = resolve_eval_seed(exp_guess, train_seed)

    cfg = load_experiment_config(
        args.config,
        step_number=step,
        n_obs=n_obs,
        noise_sigma=sigma,
    )
    key = normalize_method_key(args.method)
    if key not in ("dad", "rl_sboed", "moe_sboed", "matched_dense"):
        raise SystemExit(
            f"train only supports dad|rl_sboed|moe_sboed|matched_dense, got {args.method!r}"
        )

    exp_dir = _resolve_exp_dir(
        cfg, exp_type, args.exp_dir, create_new=False
    )
    prev_methods = load_run_config_doc(exp_dir).get("methods")
    train_record_methods = list(prev_methods) if prev_methods else [key]

    if exp_type == "objective_based":
        ctx = build_context_from_config(
            cfg,
            ensure_bank=True,
            smoke=args.smoke,
            out_dir=exp_dir,
            experiment_type=exp_type,
        )
        write_run_config(
            exp_dir,
            cfg,
            ctx.data_dir,
            experiment_type=exp_type,
            extra=_run_record_extra(
                args, methods=train_record_methods, N_obs=ctx.n_obs, seed=train_seed
            ),
        )
        display = method_display_name(key)
        print(
            f"[train] type={exp_type} {display} "
            f"config={cfg.config_path} N_obs={ctx.n_obs} exp_dir={exp_dir}"
        )
        from src.objectives.mocu.train import train_policy

        result = train_policy(
            ctx, method=display, seed=int(train_seed), smoke=args.smoke
        )
        print(
            f"[train] {display} "
            f"{float((result or {}).get('elapsed_seconds') or 0.0):.0f}s"
        )
        print(f"EXP_DIR={exp_dir}")
        return

    if _use_vector_eig_pipeline(cfg, exp_type, int(n_obs)):
        from src.objectives.eig.vector import train_vector_eig_policy

        ctx = build_context_from_config(
            cfg,
            ensure_bank=True,
            smoke=args.smoke,
            out_dir=exp_dir,
            experiment_type=exp_type,
        )
        write_run_config(
            exp_dir,
            cfg,
            ctx.data_dir,
            experiment_type=exp_type,
            extra=_run_record_extra(
                args,
                methods=train_record_methods,
                seed=train_seed,
                observation_model=(
                    "continuous_duration_max_rocof"
                    if int(ctx.n_obs) == 0
                    else "sampled_delta_f_vector"
                ),
                N_obs=ctx.n_obs,
            ),
        )
        if key == "dad":
            result = train_vector_eig_policy(
                ctx, method="dad_eig", smoke=bool(args.smoke), seed=int(train_seed)
            )
        elif key == "rl_sboed":
            result = train_vector_eig_policy(
                ctx,
                method="rl_sboed_eig",
                smoke=bool(args.smoke),
                seed=int(train_seed),
            )
        elif key == "moe_sboed":
            result = train_vector_eig_policy(
                ctx, method="moe_sboed", smoke=bool(args.smoke), seed=int(train_seed)
            )
        else:
            result = train_vector_eig_policy(
                ctx, method="matched_dense", smoke=bool(args.smoke), seed=int(train_seed)
            )
        print(
            f"[train] {result.get('method')} "
            f"val_eig={result.get('best_validation_terminal_eig')} "
            f"uniq={result.get('best_validation_n_unique_sequences')} "
            f"stage={result.get('best_stage')} "
            f"{float(result.get('elapsed_seconds') or 0.0):.0f}s "
            f"→ {result.get('checkpoint')}"
        )
        print(f"EXP_DIR={exp_dir}")
        return

    if key == "rl_sboed":
        from src.objectives.eig.pipeline import train_rl_sboed_eig
        from src.layout import load_experiment_run, load_run_config_doc

        if not load_run_config_doc(exp_dir):
            raise SystemExit(
                f"eig_based experiment missing at {exp_dir}. Run data generation first."
            )
        run = load_experiment_run(exp_dir, repo_root())
        print(f"[train] type={exp_type} EIG-specific RL-sBOED → {exp_dir}")
        path = train_rl_sboed_eig(
            run, smoke=bool(args.smoke), seed=int(train_seed)
        )
        print(
            json.dumps(
                {"experiment_type": exp_type, "policy": str(path)}, indent=2
            )
        )
        print(f"EXP_DIR={exp_dir}")
        return

    if key == "moe_sboed":
        raise SystemExit(
            "moe_sboed (BeliefConditionedMoE) is trained on the MOCU path "
            "(`--experiment_type objective_based`) or the vector-EIG path "
            "(eig_based with N_obs>0 or continuous-duration). "
            "This table-EIG entrypoint has no MoE trainer."
        )

    from src.objectives.eig.pipeline import train_dad_policy
    from src.layout import load_experiment_run, load_run_config_doc

    if not load_run_config_doc(exp_dir):
        raise SystemExit(
            f"eig_based experiment missing at {exp_dir}. "
            f"Run: ./scripts/data_generation.sh --config {args.config} "
            f"--experiment-type eig_based"
        )
    run = load_experiment_run(exp_dir, repo_root())
    print(f"[train] type={exp_type} EIG-based DAD → {exp_dir}")
    path = train_dad_policy(run, method_name="dad", smoke=bool(args.smoke))
    print(json.dumps({"experiment_type": exp_type, "policy": str(path)}, indent=2))
    print(f"EXP_DIR={exp_dir}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg, exp_type, exp_dir, identity, eval_seed = _evaluate_run_identity(args)
    method_keys = list(identity["methods"])

    if exp_type == "objective_based":
        ctx = build_context_from_config(
            cfg,
            ensure_bank=True,
            smoke=args.smoke,
            out_dir=exp_dir,
            experiment_type=exp_type,
        )
        meta = context_report_meta(ctx)
        write_run_config(
            exp_dir,
            cfg,
            ctx.data_dir,
            experiment_type=exp_type,
            extra=_run_record_extra(
                args,
                methods=method_keys,
                seed=eval_seed,
                eval_seed=eval_seed,
                smoke=bool(args.smoke),
                eval_meta=meta,
            ),
        )
        print(
            f"[evaluate] type={exp_type} methods={method_keys} "
            f"eval_seed={eval_seed} N_obs={ctx.n_obs} "
            f"mode={ctx.observation_mode} exp_dir={exp_dir}"
        )
        from src.objectives.mocu.evaluate import run_full_evaluation

        eval_meta = run_full_evaluation(
            ctx,
            methods=method_keys,
            smoke=args.smoke,
            skip_cuda_safety=bool(args.smoke),
            eval_seed=eval_seed,
        )
        summary_path = write_objective_summary_md(
            ctx.out_dir,
            system=ctx.system,
            eval_meta={
                **eval_meta,
                "T": int(cfg.step_number),
                "config_path": str(ctx.cfg.config_path),
            },
        )
        print(f"Summary → {summary_path}")
        print(f"Done → {ctx.out_dir}")
        print(f"EXP_DIR={exp_dir}")
        return

    n_obs = int(dict(cfg.raw.get("observation") or {}).get("N_obs", 0))
    if _use_vector_eig_pipeline(cfg, exp_type, n_obs):
        from src.objectives.eig.vector import evaluate_vector_eig

        vector_methods = tuple(
            VECTOR_EIG_METHOD_MAP[key]
            for key in method_keys
            if key in VECTOR_EIG_METHOD_MAP
        )
        ctx = build_context_from_config(
            cfg,
            ensure_bank=True,
            smoke=args.smoke,
            out_dir=exp_dir,
            experiment_type=exp_type,
        )
        write_run_config(
            exp_dir,
            cfg,
            ctx.data_dir,
            experiment_type=exp_type,
            extra=_run_record_extra(
                args,
                methods=list(method_keys),
                seed=eval_seed,
                eval_seed=eval_seed,
                observation_model=(
                    "continuous_duration_max_rocof"
                    if int(ctx.n_obs) == 0
                    else "sampled_delta_f_vector"
                ),
                N_obs=ctx.n_obs,
            ),
        )
        print(
            f"[evaluate] type={exp_type} methods={method_keys} "
            f"eval_seed={eval_seed} N_obs={ctx.n_obs} exp_dir={exp_dir}"
        )
        result = evaluate_vector_eig(
            ctx,
            smoke=bool(args.smoke),
            methods=vector_methods,
            eval_seed=eval_seed,
        )
        for row in result.get("summaries") or []:
            print(
                f"[evaluate] {row.get('method')} "
                f"eig={float(row.get('terminal_eig_mean', 0.0)):.4f} "
                f"uniq={row.get('n_unique_sequences')} "
                f"eval_seed={row.get('eval_seed', eval_seed)}"
            )
        print(f"EXP_DIR={exp_dir}")
        return

    from src.objectives.eig.pipeline import print_experiment_banner, run_evaluation
    from src.layout import load_experiment_run, load_run_config_doc

    if not load_run_config_doc(exp_dir):
        raise SystemExit(
            f"eig_based experiment missing at {exp_dir}. "
            f"Run data_generation + train with --experiment-type eig_based first."
        )
    run = load_experiment_run(exp_dir, repo_root())
    print(
        f"[evaluate] type={exp_type} table-EIG methods={method_keys} exp={exp_dir}"
    )
    print_experiment_banner(
        run.cfg, run.exp_dir, run.data_path,
        run.train_systems, run.test_systems, method_keys,
    )
    run_evaluation(run, methods=method_keys)
    print(f"EXP_DIR={exp_dir}")


def _diagnostic_context(args: argparse.Namespace):
    cfg = load_experiment_config(
        args.config,
        step_number=args.step_number,
        n_obs=args.n_obs,
        noise_sigma=args.noise_sigma,
    )
    exp_type = resolve_experiment_type(args.experiment_type)
    if exp_type != "objective_based":
        raise SystemExit("Policy diagnostics only support objective_based experiments")
    exp_dir = _resolve_exp_dir(cfg, exp_type, args.exp_dir, create_new=False)
    ctx = build_context_from_config(
        cfg,
        ensure_bank=True,
        smoke=False,
        out_dir=exp_dir,
        experiment_type=exp_type,
    )
    return ctx, exp_dir


def cmd_diagnose_collapse(args: argparse.Namespace) -> None:
    from src.objectives.mocu.diagnostics import (
        diagnose_conditional_action_diversity,
    )

    ctx, exp_dir = _diagnostic_context(args)
    report = diagnose_conditional_action_diversity(
        ctx,
        method=method_display_name(normalize_method_key(args.method)),
        n_rollouts=int(args.rollouts),
        seed=int(args.seed),
        device=str(args.device),
    )
    print(json.dumps(report, indent=2))
    print(f"EXP_DIR={exp_dir}")


def cmd_moe_mechanism(args: argparse.Namespace) -> None:
    exp_type = resolve_experiment_type(args.experiment_type)
    if exp_type == "eig_based":
        cfg = load_experiment_config(
            args.config,
            step_number=args.step_number,
            n_obs=args.n_obs,
            noise_sigma=args.noise_sigma,
        )
        exp_dir = _resolve_exp_dir(cfg, exp_type, args.exp_dir, create_new=False)
        ctx = build_context_from_config(
            cfg,
            ensure_bank=True,
            smoke=False,
            out_dir=exp_dir,
            experiment_type=exp_type,
        )
        from src.objectives.eig.vector import diagnose_vector_eig_moe

        report = diagnose_vector_eig_moe(
            ctx,
            n_rollouts=int(args.rollouts),
            device_name=str(args.device),
        )
        print(json.dumps(report, indent=2))
        print(f"EXP_DIR={exp_dir}")
        return
    from src.objectives.mocu.moe_diagnostics import moe_mechanism_report

    ctx, exp_dir = _diagnostic_context(args)
    report = moe_mechanism_report(
        ctx,
        n_rollouts=int(args.rollouts),
        seed=int(args.seed),
        device=str(args.device),
    )
    print(json.dumps(report, indent=2))
    print(f"EXP_DIR={exp_dir}")


def cmd_step_dad(args: argparse.Namespace) -> None:
    from src.objectives.mocu.step_dad import StepDADConfig, step_dad_report

    ctx, exp_dir = _diagnostic_context(args)
    report = step_dad_report(
        ctx,
        n_rollouts=int(args.rollouts),
        config=StepDADConfig(
            refinement_steps=int(args.refinement_steps),
            fantasy_rollouts=int(args.fantasy_rollouts),
            learning_rate=float(args.learning_rate),
            refine_from_step=int(args.refine_from_step),
            seed=int(args.seed),
            device=str(args.device),
        ),
        skip_cuda_safety=bool(args.smoke),
    )
    print(json.dumps(report, indent=2))
    print(f"EXP_DIR={exp_dir}")


def cmd_distill_myopic(args: argparse.Namespace) -> None:
    from src.objectives.mocu.diagnostics import (
        DistillationConfig,
        distill_myopic_policy,
    )

    ctx, exp_dir = _diagnostic_context(args)
    report = distill_myopic_policy(
        ctx,
        DistillationConfig(
            epochs=int(args.epochs),
            train_rollouts=int(args.train_rollouts),
            validation_rollouts=int(args.validation_rollouts),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            seed=int(args.seed),
            device=str(args.device),
        ),
    )
    print(json.dumps(report, indent=2))
    print(f"EXP_DIR={exp_dir}")


def cmd_bank_structure_audit(args: argparse.Namespace) -> None:
    """Plan-2: audit design redundancy and adaptive room before MoE work."""
    from src.banks.audit import (
        run_bank_structure_audit,
        write_audit_report,
    )

    cfg = load_experiment_config(
        args.config,
        step_number=args.step_number,
        n_obs=args.n_obs,
        noise_sigma=args.noise_sigma,
    )
    exp_type = resolve_experiment_type(args.experiment_type)
    exp_dir = _resolve_exp_dir(
        cfg, exp_type, args.exp_dir, create_new=args.exp_dir is None
    )
    ensure_result_layout(exp_dir)
    bq = dict(cfg.raw.get("bank_quality") or {})
    report = run_bank_structure_audit(
        cfg,
        n_obs=int(args.n_obs),
        noise_sigma=float(args.noise_sigma),
        support_size=int(args.support_size),
        n_outer=int(args.n_outer),
        n_inner=int(args.n_inner),
        top_k=int(args.top_k),
        seed=int(args.seed),
        near_dup_corr=float(args.near_dup_corr),
        near_dup_frac_limit=float(args.near_dup_frac_limit),
        min_fixed_advantage=float(bq.get("min_fixed_advantage", 0.01)),
        min_distinct_second_actions=int(bq.get("min_distinct_second_actions", 2)),
        min_mean_branch_value=float(bq.get("min_mean_branch_value", 0.01)),
        max_mode_second_action_prob=float(bq.get("max_mode_second_action_prob", 0.75)),
        structure_audit_horizons=list(bq.get("structure_audit_horizons") or [2, 3, 4]),
        min_gap_improve_per_horizon=float(bq.get("min_gap_improve_per_horizon", 0.005)),
        max_fixed_subsets=int(bq.get("structure_audit_max_fixed_subsets", 220)),
    )
    out = Path(exp_dir) / "diagnostics"
    json_path, md_path = write_audit_report(report, out)
    print(json.dumps(report, indent=2))
    print(f"[bank-structure-audit] wrote {json_path}")
    print(f"[bank-structure-audit] wrote {md_path}")
    print(f"EXP_DIR={exp_dir}")
    # Explicit --bank-structure-audit always enforces trap / adaptive / monotone.
    # Normal runs skip these unless bank_quality.require_* is turned on in YAML.
    fails = []
    if not report.get("myopic_beatable"):
        fails.append("myopic_trap")
    if not report.get("adaptive_room"):
        fails.append("adaptive_room")
    if not report.get("monotone_adaptive_room"):
        fails.append("monotone_adaptive_room")
    if fails:
        raise SystemExit(
            "[bank-structure-audit] FAILED "
            f"{fails} — verdict={report.get('verdict')} "
            f"myopic_beatable={report.get('myopic_beatable')} "
            f"adaptive_room={report.get('adaptive_room')} "
            f"monotone_adaptive_room={report.get('monotone_adaptive_room')}. "
            "Pass/fail only (no data filtering). Retune YAML generation params "
            "(probe_durations, contingency, …) and regenerate with --force. "
            "After checks pass, omit --bank-structure-audit for result runs."
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="experiment")
    sub = p.add_subparsers(dest="command", required=True)

    alloc = sub.add_parser(
        "allocate-dir",
        help="Create stamped result folder and print its path",
    )
    alloc.add_argument("--config", "-c", required=True)
    _add_experiment_type(alloc)
    _add_exp_dir(alloc)
    _add_T(alloc)
    _add_observation_overrides(alloc)
    alloc.set_defaults(func=cmd_allocate_dir)

    gen = sub.add_parser("generate-data", help="Generate physical / Foster banks")
    gen.add_argument("--config", "-c", required=True)
    _add_experiment_type(gen)
    _add_exp_dir(gen)
    _add_T(gen)
    _add_observation_overrides(gen)
    _add_seed(gen)
    gen.add_argument("--smoke", action="store_true")
    gen.add_argument(
        "--method",
        "-m",
        default=None,
        help=(
            "Optional comma-separated method keys for this experiment "
            f"({', '.join(EXTENDED_METHOD_KEYS)}). "
            "Omit = experiment.methods in yaml."
        ),
    )
    gen.add_argument(
        "--force",
        action="store_true",
        help="Delete and regenerate the physical bank even if complete",
    )
    gen.set_defaults(func=cmd_generate_data)

    tr = sub.add_parser("train", help="Train DAD/RL-sBOED or prepare MoE-sBOED")
    tr.add_argument("--config", "-c", required=True)
    tr.add_argument(
        "--method",
        "-m",
        required=True,
        choices=(
            "dad",
            "rl_sboed",
            "moe_sboed",
            "matched_dense",
            "DAD",
            "RL-sBOED",
            "MoE-sBOED",
            "MatchedDense",
        ),
    )
    _add_experiment_type(tr)
    _add_exp_dir(tr)
    _add_T(tr)
    _add_observation_overrides(tr)
    tr.add_argument("--seed", type=int, default=101)
    tr.add_argument("--smoke", action="store_true")
    tr.set_defaults(func=cmd_train)

    ev = sub.add_parser("evaluate", help="Evaluate methods")
    ev.add_argument("--config", "-c", required=True)
    ev.add_argument(
        "--method",
        "-m",
        default=None,
        help=(
            "Optional comma-separated method keys "
            f"({', '.join(EXTENDED_METHOD_KEYS)}). "
            "Omit = experiment.methods in yaml."
        ),
    )
    _add_experiment_type(ev)
    _add_exp_dir(ev)
    _add_T(ev)
    _add_observation_overrides(ev)
    _add_seed(ev)
    ev.add_argument("--smoke", action="store_true")
    ev.set_defaults(func=cmd_evaluate)

    collapse = sub.add_parser(
        "diagnose-collapse",
        help="Measure deterministic conditional action diversity",
    )
    collapse.add_argument("--config", "-c", required=True)
    collapse.add_argument(
        "--method", "-m", default="dad", choices=("dad", "rl_sboed", "DAD", "RL-sBOED")
    )
    _add_experiment_type(collapse)
    _add_exp_dir(collapse)
    _add_T(collapse)
    _add_observation_overrides(collapse)
    collapse.add_argument("--rollouts", type=int, default=128)
    collapse.add_argument("--seed", type=int, default=101)
    collapse.add_argument("--device", default="cpu", help="cpu|auto|cuda")
    collapse.set_defaults(func=cmd_diagnose_collapse)

    mech = sub.add_parser(
        "moe-mechanism",
        help="MoE router/expert specialization diagnostics (belief-regime evidence)",
    )
    mech.add_argument("--config", "-c", required=True)
    _add_experiment_type(mech)
    _add_exp_dir(mech)
    _add_T(mech)
    _add_observation_overrides(mech)
    mech.add_argument("--rollouts", type=int, default=128)
    mech.add_argument("--seed", type=int, default=101)
    mech.add_argument("--device", default="auto", help="cpu|auto|cuda")
    mech.set_defaults(func=cmd_moe_mechanism)

    sdad = sub.add_parser(
        "step-dad",
        help="Evaluate the semi-amortized Step-DAD baseline (refines trained DAD)",
    )
    sdad.add_argument("--config", "-c", required=True)
    _add_experiment_type(sdad)
    _add_exp_dir(sdad)
    _add_T(sdad)
    _add_observation_overrides(sdad)
    sdad.add_argument("--rollouts", type=int, default=48)
    sdad.add_argument("--refinement-steps", type=int, default=4)
    sdad.add_argument("--fantasy-rollouts", type=int, default=16)
    sdad.add_argument("--learning-rate", type=float, default=3e-4)
    sdad.add_argument("--refine-from-step", type=int, default=1)
    sdad.add_argument("--seed", type=int, default=101)
    sdad.add_argument("--device", default="auto", help="cpu|auto|cuda")
    sdad.add_argument("--smoke", action="store_true")
    sdad.set_defaults(func=cmd_step_dad)

    distill = sub.add_parser(
        "distill-myopic",
        help="Behaviorally clone the myopic baseline with the DAD architecture",
    )
    distill.add_argument("--config", "-c", required=True)
    _add_experiment_type(distill)
    _add_exp_dir(distill)
    _add_T(distill)
    _add_observation_overrides(distill)
    distill.add_argument("--epochs", type=int, default=50)
    distill.add_argument("--train-rollouts", type=int, default=512)
    distill.add_argument("--validation-rollouts", type=int, default=128)
    distill.add_argument("--batch-size", type=int, default=256)
    distill.add_argument("--learning-rate", type=float, default=3e-4)
    distill.add_argument("--weight-decay", type=float, default=1e-5)
    distill.add_argument("--seed", type=int, default=101)
    distill.add_argument("--device", default="auto", help="cpu|auto|cuda")
    distill.set_defaults(func=cmd_distill_myopic)

    audit = sub.add_parser(
        "bank-structure-audit",
        help=(
            "Plan-2: audit design redundancy and T=2 adaptive room "
            "(DAD/RL before MoE)"
        ),
    )
    audit.add_argument("--config", "-c", required=True)
    _add_experiment_type(audit)
    _add_exp_dir(audit)
    _add_T(audit)
    _add_observation_overrides(audit)
    audit.add_argument("--support-size", type=int, default=96)
    audit.add_argument("--n-outer", type=int, default=24)
    audit.add_argument("--n-inner", type=int, default=16)
    audit.add_argument("--top-k", type=int, default=12)
    audit.add_argument("--seed", type=int, default=20260808)
    audit.add_argument("--near-dup-corr", type=float, default=0.98)
    audit.add_argument("--near-dup-frac-limit", type=float, default=0.25)
    audit.set_defaults(func=cmd_bank_structure_audit)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except BankQualityError as exc:
        raise SystemExit(f"[bank-quality] FAILED\n{exc}") from exc


if __name__ == "__main__":
    main()
