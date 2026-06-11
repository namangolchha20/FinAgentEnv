"""Deterministic task graders. Each returns a score in (0, 1).

Scores are clamped to [EPS, 1 - EPS] so downstream rounding never
produces exactly 0.00 or 1.00.
"""
from .finance_engine import compute_net_worth

EPS = 1e-9


def _clamp(raw: float) -> float:
    return max(EPS, min(1.0 - EPS, raw))


def grade_debt_trap(env) -> float:
    state = env.state()
    debt_cc = state["debt"].credit_card
    raw = max(0, (35000 - debt_cc) / 35000) * 0.7
    if state["emergency_fund"] >= 10000:
        raw += 0.3
    return _clamp(raw)


def grade_balanced_growth(env) -> float:
    state = env.state()
    net = compute_net_worth(state)
    net_gain = max(0, net - 20000) / 50000
    debt_total = state["debt"].credit_card + state["debt"].personal_loan
    debt_score = max(0, (50000 - debt_total) / 50000) * 0.5
    credit_score = max(0, (state["credit_score"] - 600) / 250) * 0.2
    raw = net_gain * 0.3 + debt_score + credit_score
    return _clamp(raw)


def grade_adversarial_crash(env) -> float:
    state = env.state()
    net = compute_net_worth(state)
    net_score = 0.4 if net > 0 else 0.0
    required_ef = 3 * state["fixed_expenses"]
    ef_score = min(1.0, state["emergency_fund"] / required_ef) * 0.4
    credit_score = 0.2 if state["credit_score"] >= 650 else 0.0
    raw = net_score + ef_score + credit_score
    return _clamp(raw)


GRADERS = {
    "debt_trap": grade_debt_trap,
    "balanced_growth": grade_balanced_growth,
    "adversarial_crash": grade_adversarial_crash,
}


def grade(task_id: str, env) -> float:
    """Grade an episode; unknown task ids fall back to balanced_growth."""
    return GRADERS.get(task_id, grade_balanced_growth)(env)


def grade_breakdown(task_id: str, env) -> dict:
    """Score plus per-component breakdown (used by the dashboard)."""
    state = env.state()
    net = compute_net_worth(state)
    debt_total = state["debt"].credit_card + state["debt"].personal_loan

    if task_id == "debt_trap":
        components = [
            {"label": "Credit card debt reduced",
             "score": max(0, (35000 - state["debt"].credit_card) / 35000) * 0.7,
             "max": 0.7},
            {"label": "Emergency fund >= 10k",
             "score": 0.3 if state["emergency_fund"] >= 10000 else 0.0,
             "max": 0.3},
        ]
    elif task_id == "adversarial_crash":
        required_ef = 3 * state["fixed_expenses"]
        components = [
            {"label": "Positive net worth",
             "score": 0.4 if net > 0 else 0.0, "max": 0.4},
            {"label": "Emergency fund coverage",
             "score": min(1.0, state["emergency_fund"] / required_ef) * 0.4,
             "max": 0.4},
            {"label": "Credit score >= 650",
             "score": 0.2 if state["credit_score"] >= 650 else 0.0, "max": 0.2},
        ]
    else:
        components = [
            {"label": "Net worth growth",
             "score": min(1.0, max(0, net - 20000) / 50000) * 0.3, "max": 0.3},
            {"label": "Debt management",
             "score": max(0, (50000 - debt_total) / 50000) * 0.5, "max": 0.5},
            {"label": "Credit score",
             "score": max(0, min(1.0, (state["credit_score"] - 600) / 250)) * 0.2,
             "max": 0.2},
        ]

    return {
        "task_id": task_id,
        "score": grade(task_id, env),
        "components": [
            {**c, "score": round(c["score"], 4)} for c in components
        ],
        "net_worth": net,
    }
