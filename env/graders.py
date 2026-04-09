from .finance_engine import compute_net_worth

EPS = 1e-9

def grade_debt_trap(env) -> float:
    state = env.state
    debt_cc = state["debt"].credit_card
    raw = max(0, (35000 - debt_cc) / 35000) * 0.7
    if state["emergency_fund"] >= 10000:
        raw += 0.3
    # Clamp to [EPS, 1-EPS]
    return max(EPS, min(1.0 - EPS, raw))

def grade_balanced_growth(env) -> float:
    state = env.state
    net = compute_net_worth(state)
    net_gain = max(0, net - 20000) / 50000
    debt_total = state["debt"].credit_card + state["debt"].personal_loan
    debt_score = max(0, (50000 - debt_total) / 50000) * 0.5
    credit_score = max(0, (state["credit_score"] - 600) / 250) * 0.2
    raw = net_gain * 0.3 + debt_score + credit_score
    return max(EPS, min(1.0 - EPS, raw))

def grade_adversarial_crash(env) -> float:
    state = env.state
    net = compute_net_worth(state)
    net_score = 0.4 if net > 0 else 0.0
    required_ef = 3 * state["fixed_expenses"]
    ef_score = min(1.0, state["emergency_fund"] / required_ef) * 0.4
    credit_score = 0.2 if state["credit_score"] >= 650 else 0.0
    raw = net_score + ef_score + credit_score
    return max(EPS, min(1.0 - EPS, raw))
