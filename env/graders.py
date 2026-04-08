from .finance_engine import compute_net_worth

def grade_debt_trap(env) -> float:
    """Easy task: focus on reducing high credit card debt."""
    state = env.state
    debt_cc = state["debt"].credit_card
    # Starting debt ~35000, target below 10000
    score = max(0, (35000 - debt_cc) / 35000) * 0.7
    # Bonus if emergency fund built
    if state["emergency_fund"] >= 10000:
        score += 0.3
    return min(1.0, score)

def grade_balanced_growth(env) -> float:
    """Medium task: net worth growth and debt management."""
    state = env.state
    net = compute_net_worth(state)
    # Start net worth approx 20000 (savings + investments - debt)
    net_gain = max(0, net - 20000) / 50000
    debt_total = state["debt"].credit_card + state["debt"].personal_loan
    debt_score = max(0, (50000 - debt_total) / 50000) * 0.5
    credit_score = max(0, (state["credit_score"] - 600) / 250) * 0.2
    score = net_gain * 0.3 + debt_score + credit_score
    return min(1.0, score)

def grade_adversarial_crash(env) -> float:
    """Hard task: survive bear market and negative events."""
    state = env.state
    # Maintain positive net worth and emergency fund
    net = compute_net_worth(state)
    net_score = 0.4 if net > 0 else 0.0
    # Emergency fund >= 3 months expenses
    required_ef = 3 * state["fixed_expenses"]
    ef_score = min(1.0, state["emergency_fund"] / required_ef) * 0.4
    # Credit score above 650
    credit_score = 0.2 if state["credit_score"] >= 650 else 0.0
    score = net_score + ef_score + credit_score
    return min(1.0, score)