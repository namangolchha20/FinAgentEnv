"""Pure simulation logic for FinAgentEnv.

Every function mutates the plain-dict ``state`` in place. The
:class:`~env.environment.FinAgentEnv` orchestrates these into a
monthly pipeline. All constants come from :mod:`env.config`.
"""
import random

from .models import Action
from . import config as cfg


# ------------------------------------------------------------------ helpers

def _charge(state: dict, amount: float) -> None:
    """Charge an expense: savings first, then emergency fund, then the
    remainder goes on the credit card (real households borrow when broke)."""
    from_savings = min(amount, state["savings"])
    state["savings"] -= from_savings
    amount -= from_savings

    if amount > 0:
        from_ef = min(amount, state["emergency_fund"])
        state["emergency_fund"] -= from_ef
        amount -= from_ef

    if amount > 0:
        state["debt"].credit_card += amount
        state["borrowed_this_month"] = True


def compute_net_worth(state: dict) -> float:
    inv = state["investments"]
    debt = state["debt"]
    return (state["savings"] + state["emergency_fund"] +
            inv.stocks + inv.crypto + inv.bonds + inv.fd +
            inv.mutual_funds + inv.commodities + inv.real_estate -
            debt.credit_card - debt.personal_loan)


# ------------------------------------------------------------------ actions

_INVEST_ASSETS = ("stocks", "crypto", "bonds", "fd", "mutual_funds", "commodities")


def apply_action(state: dict, action: Action) -> tuple[bool, str]:
    """Apply the agent's action. Returns (ok, message).

    Invalid or unaffordable actions leave the state untouched and
    return ``ok=False`` so the environment can penalize them.
    """
    kind = action.action_type
    amt = max(0.0, float(action.amount or 0.0))

    if kind == "hold":
        return True, "Held position; no action taken."

    if kind == "pay_credit_card":
        payment = min(amt, state["savings"], state["debt"].credit_card)
        if payment <= 0:
            return False, "No credit card payment possible (check amount, savings, and balance)."
        state["debt"].credit_card -= payment
        state["savings"] -= payment
        state["paid_debt_this_month"] = True
        return True, f"Paid {payment:,.0f} toward credit card."

    if kind == "pay_personal_loan":
        payment = min(amt, state["savings"], state["debt"].personal_loan)
        if payment <= 0:
            return False, "No loan payment possible (check amount, savings, and balance)."
        state["debt"].personal_loan -= payment
        state["savings"] -= payment
        state["paid_debt_this_month"] = True
        return True, f"Paid {payment:,.0f} toward personal loan."

    for asset in _INVEST_ASSETS:
        if kind == f"invest_{asset}":
            if amt <= 0:
                return False, "Investment amount must be positive."
            if state["savings"] < amt:
                return False, f"Insufficient savings to invest {amt:,.0f}."
            state["savings"] -= amt
            setattr(state["investments"], asset,
                    getattr(state["investments"], asset) + amt)
            return True, f"Invested {amt:,.0f} in {asset.replace('_', ' ')}."
        if kind == f"sell_{asset}":
            held = getattr(state["investments"], asset)
            sale = min(amt, held) if amt > 0 else held
            if sale <= 0:
                return False, f"No {asset.replace('_', ' ')} holdings to sell."
            setattr(state["investments"], asset, held - sale)
            state["savings"] += sale
            return True, f"Sold {sale:,.0f} of {asset.replace('_', ' ')}."

    if kind == "buy_real_estate":
        price = amt if amt > 0 else cfg.MIN_PROPERTY_PRICE
        if price < cfg.MIN_PROPERTY_PRICE:
            return False, f"Properties start at {cfg.MIN_PROPERTY_PRICE:,.0f}."
        if state["savings"] < price:
            return False, f"Insufficient savings to buy a {price:,.0f} property."
        state["savings"] -= price
        state["investments"].real_estate += price
        return True, f"Bought real estate worth {price:,.0f}. It now earns rent (minus maintenance)."

    if kind == "sell_real_estate":
        held = state["investments"].real_estate
        sale = min(amt, held) if amt > 0 else held
        if sale <= 0:
            return False, "No real estate holdings to sell."
        proceeds = sale * (1 - cfg.PROPERTY_SALE_FEE)
        state["investments"].real_estate = held - sale
        state["savings"] += proceeds
        return True, (f"Sold real estate worth {sale:,.0f} "
                      f"(received {proceeds:,.0f} after {cfg.PROPERTY_SALE_FEE:.0%} fee).")

    if kind == "build_emergency_fund":
        move = min(amt, state["savings"])
        if move <= 0:
            return False, "Nothing transferred to emergency fund."
        state["savings"] -= move
        state["emergency_fund"] += move
        return True, f"Moved {move:,.0f} into the emergency fund."

    if kind == "withdraw_emergency_fund":
        move = min(amt, state["emergency_fund"]) if amt > 0 else state["emergency_fund"]
        if move <= 0:
            return False, "Emergency fund is empty."
        state["emergency_fund"] -= move
        state["savings"] += move
        return True, f"Withdrew {move:,.0f} from the emergency fund."

    if kind == "reduce_spending":
        cut = state["variable_expenses"] * 0.1
        new_var = max(cfg.VARIABLE_EXPENSES_FLOOR, state["variable_expenses"] - cut)
        if new_var >= state["variable_expenses"]:
            return False, "Variable expenses are already at the minimum."
        state["variable_expenses"] = new_var
        return True, f"Cut variable expenses to {new_var:,.0f}/month."

    return False, f"Unknown action '{kind}'."


# ------------------------------------------------------------------ monthly pipeline

def apply_cash_flow(state: dict) -> float:
    """Apply the month's salary and living expenses. Returns net cash flow."""
    if state.pop("income_interrupted", False):
        income = 0.0
        state["event_note"] = "No salary this month (job loss)."
    else:
        income = state["income"]

    expenses = state["fixed_expenses"] + state["variable_expenses"]
    state["savings"] += income
    _charge(state, expenses)

    # salary grows a little every month
    state["income"] *= (1 + state["income_growth"] / 12)
    return income - expenses


def apply_interest(state: dict) -> None:
    state["debt"].credit_card *= (1 + cfg.CC_MONTHLY_RATE)
    state["debt"].personal_loan *= (1 + cfg.LOAN_MONTHLY_RATE)


def simulate_market(state: dict, rng: random.Random) -> None:
    returns = cfg.MARKET_RETURNS[state["market_regime"]]
    inv = state["investments"]

    stock_r = rng.uniform(*returns["stocks"])
    inv.stocks *= (1 + stock_r)
    inv.mutual_funds *= (1 + stock_r * cfg.MUTUAL_FUND_BETA)
    inv.crypto *= (1 + rng.uniform(*returns["crypto"]))
    inv.bonds *= (1 + rng.uniform(*returns["bonds"]))
    inv.commodities *= (1 + rng.uniform(*returns["commodities"]))
    inv.real_estate *= (1 + rng.uniform(*returns["real_estate"]))
    inv.fd *= (1 + cfg.FD_MONTHLY_RATE)


def apply_real_estate_cashflow(state: dict) -> None:
    """Rental income minus maintenance costs on held property."""
    value = state["investments"].real_estate
    if value <= 0:
        return
    rent = value * cfg.RENTAL_YIELD_MONTHLY
    maintenance = value * cfg.MAINTENANCE_RATE_MONTHLY
    state["savings"] += rent
    _charge(state, maintenance)


def switch_regime(state: dict, rng: random.Random) -> None:
    if rng.random() < cfg.REGIME_SWITCH_PROB:
        state["market_regime"] = rng.choice(list(cfg.REGIMES))


def apply_event(state: dict, rng: random.Random) -> None:
    table = cfg.EVENT_TABLES[state.get("event_profile", "normal")]
    weights = [row[1] for row in table]
    name, _, kind, low, high = rng.choices(table, weights=weights, k=1)[0]
    state["event"] = name

    if kind == "cost":
        _charge(state, rng.uniform(low, high))
    elif kind == "gain":
        state["savings"] += rng.uniform(low, high)
    elif kind == "income":
        state["income_interrupted"] = True
    elif kind == "raise":
        state["income"] *= (1 + rng.uniform(low, high))


def update_credit_score(state: dict) -> None:
    # keep utilization in sync with the actual revolving balance
    state["credit_used"] = state["debt"].credit_card

    util = state["credit_used"] / state["credit_limit"]
    if util > 0.8:
        state["credit_score"] -= 20
    elif util < 0.3:
        state["credit_score"] += 10

    if state.pop("paid_debt_this_month", False):
        state["credit_score"] += 5
    if state.pop("borrowed_this_month", False):
        state["credit_score"] -= 15

    state["credit_score"] = max(cfg.CREDIT_SCORE_MIN,
                                min(cfg.CREDIT_SCORE_MAX, state["credit_score"]))


# ------------------------------------------------------------------ reward & diagnostics

def compute_reward(prev_net: float, curr_net: float, state: dict,
                   action_ok: bool = True) -> float:
    reward = (curr_net - prev_net) / cfg.REWARD_NET_WORTH_SCALE
    total_debt = state["debt"].credit_card + state["debt"].personal_loan
    reward -= total_debt / cfg.REWARD_DEBT_SCALE
    reward += (state["credit_score"] - 600) / cfg.REWARD_CREDIT_SCALE
    if curr_net > prev_net:
        reward += cfg.REWARD_GROWTH_BONUS
    if state["emergency_fund"] >= cfg.EF_TARGET_MONTHS * state["fixed_expenses"]:
        reward += cfg.REWARD_EF_BONUS
    else:
        reward -= cfg.REWARD_EF_BONUS
    if state["savings"] < cfg.LOW_SAVINGS_THRESHOLD:
        reward -= cfg.REWARD_LOW_SAVINGS_PENALTY
    if not action_ok:
        reward -= cfg.REWARD_INVALID_ACTION_PENALTY
    return max(-1.0, min(1.0, reward))


def failure_analysis(state: dict) -> list:
    failures = []
    if state["debt"].credit_card > 20000:
        failures.append("high_credit_card_debt")
    if state["emergency_fund"] < cfg.EF_TARGET_MONTHS * state["fixed_expenses"]:
        failures.append("low_emergency_fund")
    if state["credit_score"] < 600:
        failures.append("poor_credit_score")
    return failures
