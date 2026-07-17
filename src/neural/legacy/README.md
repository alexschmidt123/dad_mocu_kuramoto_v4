# Neural legacy modules

| Module | Role |
|--------|------|
| `advanced_dad.py` | Normalized history encoder used by Stage-2 screens |
| `ppo_stage2.py` | Multi-encoder Stage-2 policy (H0/H1/H2 × B0/B1/B2) for diagnostics |

Production code must use:

- `src.neural.policy` — REINFORCE DAD (`DADPolicy`)
- `src.neural.train` — REINFORCE trainer
- `src.neural.rl_policy` — PPO DAD / RL-sBOED shared backbone
