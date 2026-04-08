import random
from .models import Debt, Investments, Action

def apply_action(state: dict, action: Action):
    amt = action.amount
    if action.action_type == "pay_credit_card":
        payment = min(amt, state["savings"])
        state["debt"].credit_card -= payment
        state["savings"] -= payment
        state["credit_used"] -= payment
    elif action.action_type == "pay_personal_loan":
        payment = min(amt, state["savings"])
        state["debt"].personal_loan -= payment
        state["savings"] -= payment
    elif action.action_type == "invest_stocks":
        if state["savings"] >= amt:
            state["savings"] -= amt
            state["investments"].stocks += amt
    elif action.action_type == "invest_crypto":
        if state["savings"] >= amt:
            state["savings"] -= amt
            state["investments"].crypto += amt
    elif action.action_type == "invest_bonds":
        if state["savings"] >= amt:
            state["savings"] -= amt
            state["investments"].bonds += amt
    elif action.action_type == "invest_fd":
        if state["savings"] >= amt:
            state["savings"] -= amt
            state["investments"].fd += amt
    elif action.action_type == "invest_mutual_funds":
        if state["savings"] >= amt:
            state["savings"] -= amt
            state["investments"].mutual_funds += amt
    elif action.action_type == "invest_commodities":
        if state["savings"] >= amt:
            state["savings"] -= amt
            state["investments"].commodities += amt
    elif action.action_type == "buy_real_estate":
        if state["savings"] >= 50000:
            state["savings"] -= 50000
            state["investments"].real_estate += 50000
    elif action.action_type == "build_emergency_fund":
        move = min(amt, state["savings"])
        state["savings"] -= move
        state["emergency_fund"] += move
    elif action.action_type == "reduce_spending":
        state["variable_expenses"] *= 0.9

def apply_interest(state: dict):
    state["debt"].credit_card *= 1.03
    state["debt"].personal_loan *= 1.01

def simulate_market(state: dict, rng: random.Random):
    regime = state["market_regime"]
    if regime == "bull":
        stock_return = rng.uniform(0.05, 0.15)
    elif regime == "bear":
        stock_return = rng.uniform(-0.2, -0.05)
    else:
        stock_return = rng.uniform(-0.05, 0.05)
    state["investments"].stocks *= (1 + stock_return)
    state["investments"].crypto *= (1 + rng.uniform(-0.3, 0.3))
    state["investments"].bonds *= (1 + rng.uniform(0.01, 0.05))
    state["investments"].fd *= 1.02
    state["investments"].mutual_funds *= (1 + stock_return * 0.8)
    state["investments"].commodities *= (1 + rng.uniform(-0.05, 0.1))
    state["investments"].real_estate *= (1 + rng.uniform(0.01, 0.05))

def apply_real_estate_income(state: dict):
    rental_income = state["investments"].real_estate * 0.01
    state["savings"] += rental_income

def switch_regime(state: dict, rng: random.Random):
    if rng.random() < 0.2:
        state["market_regime"] = rng.choice(["bull", "bear", "sideways"])

def apply_event(state: dict, rng: random.Random):
    events = [
        ("none", 0),
        ("medical", 10000),
        ("job_loss", 15000),
        ("bonus", -15000),
    ]
    event, impact = rng.choice(events)
    state["event"] = event
    state["savings"] = max(0, state["savings"] - impact)

def update_credit_score(state: dict):
    util = state["credit_used"] / state["credit_limit"]
    if util > 0.8:
        state["credit_score"] -= 20
    elif util < 0.3:
        state["credit_score"] += 10
    state["credit_score"] = max(300, min(850, state["credit_score"]))

def compute_net_worth(state: dict) -> float:
    inv = state["investments"]
    debt = state["debt"]
    return (state["savings"] + state["emergency_fund"] +
            inv.stocks + inv.crypto + inv.bonds + inv.fd +
            inv.mutual_funds + inv.commodities + inv.real_estate -
            debt.credit_card - debt.personal_loan)

def compute_reward(prev_net: float, curr_net: float, state: dict) -> float:
    reward = (curr_net - prev_net) / 20000.0
    total_debt = state["debt"].credit_card + state["debt"].personal_loan
    reward -= total_debt / 300000.0
    reward += (state["credit_score"] - 600) / 1000.0
    if curr_net > prev_net:
        reward += 0.2
    if state["emergency_fund"] < 3 * state["fixed_expenses"]:
        reward -= 0.3
    else:
        reward += 0.3
    if state["savings"] < 5000:
        reward -= 0.2
    # Clip to reasonable range
    return max(-1.0, min(1.0, reward))

def failure_analysis(state: dict) -> list:
    failures = []
    if state["debt"].credit_card > 20000:
        failures.append("high_credit_card_debt")
    if state["emergency_fund"] < 3 * state["fixed_expenses"]:
        failures.append("low_emergency_fund")
    if state["credit_score"] < 600:
        failures.append("poor_credit_score")
    return failures