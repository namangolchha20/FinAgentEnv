import random

try:
    from openenv.core.env_server import Environment
except ImportError:  # allow local use without openenv installed
    class Environment:
        pass

from .models import Observation, Action, StepResult, Debt, Investments, Info
from .tasks import get_task
from . import config as cfg
from .finance_engine import (
    apply_action, apply_cash_flow, apply_interest, simulate_market,
    apply_real_estate_cashflow, switch_regime, apply_event,
    update_credit_score, compute_net_worth, compute_reward, failure_analysis,
)


class FinAgentEnv(Environment):
    def __init__(self):
        self.max_months = cfg.MAX_MONTHS
        self._rng = None
        self._state = None

    # ------------------------------------------------------------- API

    def reset(self, task_id: str = None, seed: int = None) -> Observation:
        self._rng = random.Random(seed) if seed is not None else random.Random()

        task = get_task(task_id)
        init = task["initial"]

        self._state = {
            "month": 1,
            "income": cfg.STARTING_INCOME,
            "income_growth": cfg.INCOME_GROWTH_ANNUAL,
            "fixed_expenses": cfg.FIXED_EXPENSES,
            "variable_expenses": cfg.VARIABLE_EXPENSES,
            "savings": float(init["savings"]),
            "emergency_fund": float(init["emergency_fund"]),
            "debt": Debt(**init["debt"]),
            "credit_score": float(init["credit_score"]),
            "credit_limit": cfg.CREDIT_LIMIT,
            "credit_used": float(init["debt"]["credit_card"]),
            "investments": Investments(**init["investments"]),
            "market_regime": init["market_regime"],
            "event": "none",
            "event_profile": task["event_profile"],
        }
        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        if self._state is None:
            raise RuntimeError("step() called before reset()")

        s = self._state
        prev_net = compute_net_worth(s)

        # 1. agent allocates this month's money
        action_ok, action_msg = apply_action(s, action)
        # 2. salary arrives, living expenses are paid
        cash_flow = apply_cash_flow(s)
        # 3. debt accrues interest
        apply_interest(s)
        # 4. markets move
        simulate_market(s, self._rng)
        # 5. property pays rent, costs maintenance
        apply_real_estate_cashflow(s)
        # 6. life happens
        apply_event(s, self._rng)
        # 7. regime may shift for next month
        switch_regime(s, self._rng)
        # 8. credit bureau updates
        update_credit_score(s)

        s["month"] += 1

        curr_net = compute_net_worth(s)
        reward = compute_reward(prev_net, curr_net, s, action_ok)
        done = s["month"] > self.max_months

        info = Info(
            net_worth=curr_net,
            failures=failure_analysis(s),
            regime=s["market_regime"],
            event=s["event"],
            cash_flow=cash_flow,
            action_ok=action_ok,
            action_message=action_msg,
        )
        return StepResult(observation=self._make_observation(), reward=reward,
                          done=done, info=info)

    def state(self) -> dict:
        if self._state is None:
            return {}
        out = dict(self._state)
        out["debt"] = self._state["debt"].model_copy()
        out["investments"] = self._state["investments"].model_copy()
        return out

    # ------------------------------------------------------------- internals

    def _make_observation(self) -> Observation:
        s = self._state
        fields = {k: v for k, v in s.items()
                  if k in Observation.model_fields}
        return Observation(**fields, net_worth=compute_net_worth(s))
