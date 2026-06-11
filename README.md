---
title: FinAgentEnv
emoji: 💰
colorFrom: green
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# FinAgentEnv

**OpenEnv-compatible personal finance simulation** for training and evaluating AI agents.

An agent manages six monthly steps of real-world financial decisions: cash flow, debt, eight asset classes, real estate, credit score, and stochastic life events — all under shifting market regimes (`bull`, `bear`, `sideways`).

## Quick start

```bash
git clone https://github.com/namangolchha20/FinAgentEnv.git
cd FinAgentEnv
pip install -r requirements.txt
pip install -e .
openenv validate
```

Run the HTTP server (used by Hugging Face Spaces):

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

- **API docs:** http://localhost:7860/docs  
- **Demo dashboard:** http://localhost:7860 (optional human UI)

## Environment API

```python
from env import FinAgentEnv
from env.models import Action

env = FinAgentEnv()
obs = env.reset(task_id="debt_trap", seed=42)

result = env.step(Action(action_type="pay_credit_card", amount=5000))
print(result.observation.net_worth, result.reward, result.done)

state = env.state()  # full internal dict
```

| Method | Returns |
|--------|---------|
| `reset(task_id, seed)` | `Observation` — starting state for the episode |
| `step(action)` | `StepResult` — `observation`, `reward`, `done`, `info` |
| `state()` | `dict` — copy of current environment state |

### Tasks

| Task ID | Difficulty | Goal |
|---------|-----------|------|
| `debt_trap` | Easy | Reduce credit card debt, build emergency fund ≥ $10k |
| `balanced_growth` | Medium | Grow net worth while managing debt and credit |
| `adversarial_crash` | Hard | Survive bear market + harsh events with positive NW |

Graders live in `env/graders.py` and are registered in `openenv.yaml`.

### Action space (20 actions)

```json
{"action_type": "pay_credit_card", "amount": 5000}
```

| Category | Actions |
|----------|---------|
| Debt | `pay_credit_card`, `pay_personal_loan` |
| Invest | `invest_stocks`, `invest_mutual_funds`, `invest_crypto`, `invest_bonds`, `invest_fd`, `invest_commodities` |
| Sell | `sell_stocks`, `sell_mutual_funds`, `sell_crypto`, `sell_bonds`, `sell_fd`, `sell_commodities` |
| Real estate | `buy_real_estate` (min $50k), `sell_real_estate` (5% fee) |
| Safety | `build_emergency_fund`, `withdraw_emergency_fund` |
| Lifestyle | `reduce_spending`, `hold` |

Invalid or unaffordable actions are rejected (`info.action_ok = false`), state is unchanged, reward is penalized by −0.1.

### Observation

Key fields: `month`, `income`, `fixed_expenses`, `variable_expenses`, `savings`, `emergency_fund`, `net_worth`, `debt`, `credit_score`, `credit_used`, `investments`, `market_regime`, `event`.

`info` per step: `net_worth`, `cash_flow`, `action_ok`, `action_message`, `failures`, `regime`, `event`.

### Reward (dense, clipped to [−1, 1])

Net worth change, debt penalty, credit score bonus, emergency fund adequacy, low-savings penalty, invalid-action penalty. See `env/finance_engine.py` → `compute_reward`.

## HTTP API (FastAPI)

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness probe |
| `GET /tasks` | Task metadata + action types |
| `POST /reset?task_id&seed&session_id` | Start episode |
| `POST /step?session_id` | Apply action |
| `GET /state?session_id` | Current state |
| `POST /grade?session_id` | Task grader + score breakdown |

Default `session_id` is `"default"`. Each session gets its own `FinAgentEnv` instance.

## Baseline evaluator (optional)

`inference.py` runs a fixed LLM through all three tasks for benchmarking — **not training**.

```bash
export OPENAI_API_KEY=your_key
python inference.py
```

| Variable | Default |
|----------|---------|
| `OPENAI_API_KEY` / `API_KEY` | required |
| `API_BASE_URL` | `https://api.openai.com/v1` |
| `MODEL_NAME` | `gpt-4o-mini` |

## Deploy to Hugging Face Spaces

1. Create a new **Docker** Space (public).
2. Push this repo — the README frontmatter (`sdk: docker`, `app_port: 7860`) is already set.
3. The `Dockerfile` builds and starts uvicorn on port **7860**.
4. Verify: `POST /reset` returns a valid observation JSON.

```bash
docker build -t finagent .
docker run -p 7860:7860 finagent
curl -X POST "http://localhost:7860/reset?task_id=debt_trap&seed=42"
```

## Project layout

```
env/
  config.py           # simulation constants
  models.py           # Pydantic Observation / Action / StepResult
  finance_engine.py   # pure simulation logic
  environment.py      # FinAgentEnv (OpenEnv entry point)
  tasks.py            # task definitions
  graders.py          # deterministic task graders
server/app.py         # FastAPI wrapper + optional dashboard
frontend/             # optional demo UI (not required for training)
inference.py          # optional LLM baseline evaluator
openenv.yaml          # OpenEnv manifest
tests/test_env.py     # pytest suite (25 tests)
```

## OpenEnv compliance

- `reset(task_id, seed) → Observation`
- `step(action) → StepResult`
- `state() → dict`
- Typed Pydantic models
- `openenv.yaml` with three tasks and grader references
- Deterministic under a fixed seed
- Passes `openenv validate`

## Development

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
python -m pytest tests/ -q
openenv validate
```

`[dev]` adds `pytest` and `openai` (for `inference.py`).

## License

MIT — see [LICENSE](LICENSE).
