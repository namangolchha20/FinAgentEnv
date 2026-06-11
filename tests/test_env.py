import copy

import pytest

from env import config as cfg
from env.environment import FinAgentEnv
from env.finance_engine import apply_action, compute_net_worth
from env.graders import grade, grade_breakdown, GRADERS, EPS
from env.models import Action, ACTION_TYPES


def make_env(task="balanced_growth", seed=42):
    env = FinAgentEnv()
    env.reset(task_id=task, seed=seed)
    return env


# ------------------------------------------------------------------ reset / state

def test_reset_returns_observation_with_net_worth():
    env = FinAgentEnv()
    obs = env.reset(task_id="debt_trap", seed=1)
    assert obs.month == 1
    assert obs.debt.credit_card == 35000
    assert obs.net_worth == pytest.approx(compute_net_worth(env.state()))


def test_state_is_callable_and_returns_copy():
    env = make_env()
    state = env.state()
    assert isinstance(state, dict)
    state["savings"] = -999
    state["debt"].credit_card = -999
    fresh = env.state()
    assert fresh["savings"] != -999
    assert fresh["debt"].credit_card != -999


def test_step_before_reset_raises():
    env = FinAgentEnv()
    with pytest.raises(RuntimeError):
        env.step(Action(action_type="hold"))


def test_unknown_task_falls_back_to_default():
    env = FinAgentEnv()
    obs = env.reset(task_id="nope", seed=1)
    assert obs.savings == 20000


# ------------------------------------------------------------------ determinism

def test_seeded_episodes_are_reproducible():
    def run():
        env = make_env("adversarial_crash", seed=7)
        results = []
        for _ in range(cfg.MAX_MONTHS):
            r = env.step(Action(action_type="invest_stocks", amount=2000))
            results.append((round(r.reward, 8), r.observation.market_regime,
                            r.info.event, round(r.info.net_worth, 4)))
        return results

    assert run() == run()


# ------------------------------------------------------------------ cash flow

def test_income_and_expenses_applied_each_month():
    env = make_env(seed=3)
    s = env._state
    income, fixed, variable = s["income"], s["fixed_expenses"], s["variable_expenses"]
    result = env.step(Action(action_type="hold"))
    assert result.info.cash_flow == pytest.approx(income - fixed - variable)


def test_income_grows_monthly():
    env = make_env(seed=3)
    before = env._state["income"]
    env.step(Action(action_type="hold"))
    assert env._state["income"] >= before * (1 + cfg.INCOME_GROWTH_ANNUAL / 12) * 0.99


# ------------------------------------------------------------------ actions

def test_pay_credit_card_cannot_overpay():
    env = make_env(seed=1)
    s = env._state
    s["debt"].credit_card = 1000
    s["savings"] = 50000
    ok, _ = apply_action(s, Action(action_type="pay_credit_card", amount=99999))
    assert ok
    assert s["debt"].credit_card == 0
    assert s["savings"] == 49000


def test_invest_requires_sufficient_savings():
    env = make_env(seed=1)
    s = env._state
    s["savings"] = 100
    snapshot = copy.deepcopy({k: v for k, v in s.items()})
    ok, msg = apply_action(s, Action(action_type="invest_stocks", amount=5000))
    assert not ok
    assert s["savings"] == snapshot["savings"]
    assert s["investments"].stocks == snapshot["investments"].stocks


def test_sell_caps_at_holdings_and_zero_means_all():
    env = make_env(seed=1)
    s = env._state
    s["investments"].stocks = 4000
    ok, _ = apply_action(s, Action(action_type="sell_stocks", amount=0))
    assert ok
    assert s["investments"].stocks == 0


def test_buy_real_estate_uses_amount_and_minimum():
    env = make_env(seed=1)
    s = env._state
    s["savings"] = 100_000
    ok, msg = apply_action(s, Action(action_type="buy_real_estate", amount=10_000))
    assert not ok  # below minimum price
    ok, _ = apply_action(s, Action(action_type="buy_real_estate", amount=80_000))
    assert ok
    assert s["investments"].real_estate == 80_000
    assert s["savings"] == 20_000


def test_sell_real_estate_charges_fee():
    env = make_env(seed=1)
    s = env._state
    s["investments"].real_estate = 100_000
    savings_before = s["savings"]
    ok, _ = apply_action(s, Action(action_type="sell_real_estate", amount=0))
    assert ok
    assert s["savings"] == pytest.approx(
        savings_before + 100_000 * (1 - cfg.PROPERTY_SALE_FEE))


def test_unknown_action_is_invalid_and_penalized():
    env = make_env(seed=1)
    result = env.step(Action(action_type="time_travel", amount=1))
    assert result.info.action_ok is False

    env2 = make_env(seed=1)
    baseline = env2.step(Action(action_type="hold"))
    assert result.reward <= baseline.reward


def test_all_declared_action_types_are_handled():
    for action_type in ACTION_TYPES:
        env = make_env(seed=5)
        s = env._state
        s["savings"] = 200_000
        s["investments"].stocks = 10_000
        s["investments"].crypto = 10_000
        s["investments"].bonds = 10_000
        s["investments"].fd = 10_000
        s["investments"].mutual_funds = 10_000
        s["investments"].commodities = 10_000
        s["investments"].real_estate = 60_000
        s["emergency_fund"] = 10_000
        ok, msg = apply_action(s, Action(action_type=action_type, amount=5_000))
        if action_type == "buy_real_estate":
            ok, msg = apply_action(s, Action(action_type=action_type, amount=60_000))
        assert ok, f"{action_type} should be valid: {msg}"


# ------------------------------------------------------------------ credit

def test_credit_used_stays_synced_with_card_balance():
    env = make_env("debt_trap", seed=2)
    for _ in range(3):
        env.step(Action(action_type="hold"))
    s = env._state
    assert s["credit_used"] == pytest.approx(s["debt"].credit_card)


def test_credit_score_bounded():
    env = make_env("debt_trap", seed=2)
    for _ in range(cfg.MAX_MONTHS):
        env.step(Action(action_type="hold"))
    assert cfg.CREDIT_SCORE_MIN <= env._state["credit_score"] <= cfg.CREDIT_SCORE_MAX


# ------------------------------------------------------------------ episode shape

def test_episode_ends_after_max_months():
    env = make_env(seed=9)
    done_flags = [env.step(Action(action_type="hold")).done
                  for _ in range(cfg.MAX_MONTHS)]
    assert done_flags == [False] * (cfg.MAX_MONTHS - 1) + [True]


def test_rewards_clipped():
    env = make_env("adversarial_crash", seed=11)
    for _ in range(cfg.MAX_MONTHS):
        r = env.step(Action(action_type="invest_crypto", amount=1000))
        assert -1.0 <= r.reward <= 1.0


# ------------------------------------------------------------------ graders

@pytest.mark.parametrize("task_id", list(GRADERS))
def test_graders_bounded(task_id):
    env = make_env(task_id, seed=4)
    for _ in range(cfg.MAX_MONTHS):
        env.step(Action(action_type="hold"))
    score = grade(task_id, env)
    assert EPS <= score <= 1.0 - EPS


@pytest.mark.parametrize("task_id", list(GRADERS))
def test_grade_breakdown_matches_score(task_id):
    env = make_env(task_id, seed=4)
    for _ in range(cfg.MAX_MONTHS):
        env.step(Action(action_type="build_emergency_fund", amount=5000))
    data = grade_breakdown(task_id, env)
    assert data["score"] == pytest.approx(grade(task_id, env))
    assert data["components"]


def test_debt_trap_grader_rewards_payoff():
    env_pay = make_env("debt_trap", seed=6)
    env_hold = make_env("debt_trap", seed=6)
    for _ in range(cfg.MAX_MONTHS):
        env_pay.step(Action(action_type="pay_credit_card", amount=20000))
        env_hold.step(Action(action_type="hold"))
    assert grade("debt_trap", env_pay) > grade("debt_trap", env_hold)
