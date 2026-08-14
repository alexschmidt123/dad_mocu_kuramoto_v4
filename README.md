# dad_mocu_kuramoto

Sequential Bayesian optimal experiment design for power-grid swing dynamics.
Methods choose probe actions to reduce uncertainty and minimize posterior-safe control effort `u_ctrl`.

## Environment

Use conda env **`mocu_optimized`** (Python 3.10, PyTorch 2.4.0+cu121, CUDA 12.1):

```bash
conda create -n mocu_optimized python=3.10 -y
conda activate mocu_optimized
# CUDA toolkit (for PyCUDA), then PyTorch cu121, then the rest — see requirements.txt header
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
export PYTHONPATH="$(pwd)"
```

## Hardware (development machine)

| | |
|---|---|
| Host | `ecen-80r5x04.engr.tamu.edu` |
| OS | Ubuntu 22.04.5 LTS (kernel 6.8.0-124-generic) |
| CPU | 13th Gen Intel Core i7-13700F (16 cores / 24 threads, up to 5.2 GHz) |
| RAM | 62 GiB |
| GPU | NVIDIA GeForce RTX 4090 (24 GB, compute capability 8.9) |
| Driver | 560.35.05 |

## Experimental families

Two primary families share the same method set (DAD, RL-sBOED, MoE-sBOED, Myopic, Fixed, Random):

| Family | Config | Objective | Notes |
|---|---|---|---|
| Continuous duration (IEEE5) | `configs/ieee5_continuous_duration.yaml` | MOCU / EIG | fixed amp/bus, duration-only ξ, `N_obs=0` max-RoCoF, **no reset**, chronological ξ₁<ξ₂<…ξ_T |
| SIR ODE (iDAD-style) | `configs/sir_ode.yaml` | vector EIG | chronological measurement times, `y=I(τ)+ε`, see Ivanova et al. NeurIPS 2021 App. D.6 |

```bash
# Continuous-duration power-grid (default publication path)
bash run.sh --config configs/ieee5_continuous_duration.yaml --force \
  --experiment_type objective_based -T 3 --N_obs 0 --noise_sigma 0.01 --seed 101
python3 -m src.banks.diagnose_duration --config configs/ieee5_continuous_duration.yaml

# SIR ODE EIG (T=4 and T=5)
bash run.sh --config configs/sir_ode.yaml --experiment_type eig_based -T 4 --seed 101
bash run.sh --config configs/sir_ode.yaml --experiment_type eig_based -T 5 --seed 101
```

Layout (root keeps only orchestration; domain names, not observation/bank modes):

```text
src/
  config.py            # YAML + SYSTEM_CONFIGS
  experiment.py        # sole CLI (generate / train / evaluate)
  domains/swing|sir/   # physics + design
  banks/               # bank I/O + core audits (not the on-disk folder)
  observations/        # compress / likelihood / carry-state
  control/             # u_req, posterior, rewards, myopic/fixed
  inference/           # sPCE + scoring
  policies/            # policy nets only (dad / rl_sboed / moe)
  reporting/           # run dirs + summary tables
  objectives/          # optimization goals (not shared architecture)
    mocu/              # objective_based / MOCU train+eval
    eig/               # eig_based / EIG train+eval
    # add more goals here as needed

data/                  # on-disk banks only (ieee5_*, sir_ode, …)
tools/                 # offline audits / diagnostics (optional)
tests/                 # optional only; safe to delete
```

Result folders under `experiments/` must be stamped
`MMDDYYYY_HHMMSS_<config>_Uctrl|EIG_T#_Nobs#_sigmaX`. Never create
`experiments/_…` smoke dirs, `experiments/*_logs`, or a top-level `logs/`
folder. Run logs go in `<result_folder>/logs/` (e.g. `logs/run_log.log`).
Use `/tmp` for scratch or sweep-coordinator nohup only.

Only `README.md` is the project readme. Under `documents/`, keep
`moe_sboed_workflow.txt`, `publication_experiment_plan.txt`, `sBOED_design.tex`,
and `conference_poster_report.md` (plus `papers/` / `images/` assets).

## Run

Primary power-grid config is `configs/ieee5_continuous_duration.yaml`. Runtime options are:

- `--T`: probe horizon (default **5**).
- `--N_obs`: observations retained per experiment (optional, default **0**). Aliases: `--n-obs`, `--n_obs`.
- `--noise_sigma`: observation-noise standard deviation (optional, default **0.005**). Alias: `--noise-sigma`.

Do not put `T`, `N_obs`, or `noise_sigma` in the config filename.

```bash
# Defaults: T=5, N_obs=0, noise_sigma=0.005
bash run.sh --config configs/ieee5.yaml --experiment_type objective_based

# Explicit time-series observation settings
bash run.sh --config configs/ieee5.yaml --T 3 --N_obs 120 \
  --noise_sigma 0.005 --experiment_type objective_based
```

Observation rules:

- `N_obs=0` uses the scalar `max_rocof.npy` observation; `noise_sigma` is then measured in Hz/s.
- `N_obs>0` uses `N_obs` evenly spaced samples from the stored `delta_f.npy` trajectory; `noise_sigma` is then measured in Hz.
- `N_obs` must not exceed the stored trajectory length `N_sim` (currently 1600).
- Changing `N_obs` or `noise_sigma` reuses the existing clean physical bank. Do not use `--force` only for an observation-setting change.

Sweep arguments may be comma-separated. The sweep is the Cartesian product of
`--configs` × `--T` × `--N_obs` × `--noise_sigma`:

```bash
bash sweep_run.sh --configs ieee5,ieee9,ieee14 --T 2,3,4,5 \
  --N_obs 120 --noise_sigma 0.001,0.005,0.01,0.02
```

`--force` regenerates the physical bank under `data/<system>/` (otherwise reused).
The data-generation, DAD-training, RL-SBOED-training, and evaluation scripts also
accept the same optional `--N_obs` and `--noise_sigma` arguments.

Results are stored using:

```text
MMDDYYYY_HHMMSS_<system>_<type>_T<T>_Nobs<N_obs>_sigma<sigma>
```

For example:

```text
07232026_215655_ieee14_Uctrl_T3_Nobs0_sigma0p005
```

For folder names only, `objective_based` is shortened to `Uctrl` and `eig_based`
to `EIG`. The CLI and configuration values remain `objective_based` and
`eig_based`.

After every generate/reuse (and again on train/eval load), **bank quality gates** run by default (`bank_quality` in the YAML). They check finite trajectories, enough `U>0` mass, prior headroom (`Q95(U)−mean(U)`), and max_|ROCOF|–`U` correlation. Failure raises an error and stops later stages.

### Objective-control validity

- A method is valid only when its held-out safety rate is at least **0.95**.
- Methods below 0.95 safety are reported as `INVALID` and receive no rank.
- The final objective metric is
  `mean_MOCU = mean(u_ctrl - u_ctrl_optimal)` over common held-out systems.
- Among valid methods, lower `mean_MOCU` is better; `mean_u_ctrl` is reported
  as a secondary physical-control metric.
- Valid methods are ranked by lower mean `u_ctrl`.
- The terminal rule is fixed for each IEEE system across every horizon:
  `u_ctrl = Q_0.95(U | history) + fixed_margin`, snapped to the system control
  grid.  `T` changes only the number of experiments; the rule must never be
  recalibrated separately for each `T`.
- Calibration/validation parameter draws are excluded from posterior particle
  support to prevent coverage leakage.
- DAD uses a trajectory-level REINFORCE update; RL-SBOED uses dense
  safety-aware cost-reduction rewards with PPO.

Audits and diagnostic harnesses live under `tools/` (offline scripts) or in
core `src/` when they are part of the product path (e.g. `src.banks.audit`,
`src.banks.diagnose_control`, `python -m src.experiment bank-structure-audit`).
`tests/` is optional: removing it must not affect runs. Do not write audit
artifacts into the project root, production `data/`, or stamped `experiments/`
result folders (use `tools/.../results/` or `/tmp`).

### `data/<system>/` layout

```text
data/ieee5/
  meta/
    catalog.json   # probe actions: (amplitude, bus, duration_s) per action id
    bank.yaml      # slim provenance (shapes, seeds, ode_dt, …; no time_vector)
  train/
    delta_f.npy    # (n_θ, n_actions, N_sim) clean probe-bus Δf(t) [Hz]
    max_rocof.npy  # (n_θ, n_actions) max |ROCOF|
    theta_M.npy    # (n_θ, N) inertia draws
    theta_K.npy    # (n_θ, N) coupling draws
    psi_star.npy   # (n_θ,) ψ_θ*: Yoon model-specific min safe operator
  test/
    …              # same arrays for the held-out θ set
```

`psi_star.npy` stores ψ_θ* for each particle (legacy name was `U.npy`; auto-migrated on load). Methods never see true θ; they update a posterior over these particles and choose a robust operator ψ* (IBR: max over support). Belief MOCU is then E[OCU] = ψ* − E[ψ_θ*].

Derived artifacts (frozen terminal rule, Fixed subset) live under the experiment `model/` folder — not under `data/`.

Topologies match MATPOWER [`case5`](https://github.com/MATPOWER/matpower/blob/master/data/case5.m) / [`case9`](https://github.com/MATPOWER/matpower/blob/master/data/case9.m) / [`case14`](https://github.com/MATPOWER/matpower/blob/master/data/case14.m) (UW PSTCA).
