import random
from openenv.core.env_server import Environment
from .models import Observation, Action, StepResult, Debt, Investments, Info
from .finance_engine import (
    apply_action, apply_interest, simulate_market, apply_real_estate_income,
    switch_regime, apply_event, update_credit_score, compute_net_worth,
    compute_reward, failure_analysis
)

class FinAgentEnv(Environment):
    def __init__(self):
        self.max_months = 6
        self._rng = None
        self.state = None

    def reset(self, task_id: str = None, seed: int = None) -> Observation:
        # Set up deterministic RNG
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        # Task-specific initial conditions
        if task_id == "debt_trap":
            # High credit card debt, low savings
            debt = Debt(credit_card=35000, personal_loan=20000)
            investments = Investments(stocks=0, crypto=0, bonds=0, fd=0,
                                      mutual_funds=0, commodities=0, real_estate=0)
            savings = 5000
            emergency_fund = 0
            credit_score = 580
            market_regime = "sideways"
        elif task_id == "balanced_growth":
            debt = Debt(credit_card=15000, personal_loan=10000)
            investments = Investments(stocks=5000, crypto=0, bonds=2000, fd=5000,
                                      mutual_funds=3000, commodities=2000, real_estate=0)
            savings = 20000
            emergency_fund = 5000
            credit_score = 650
            market_regime = "bull"
        elif task_id == "adversarial_crash":
            debt = Debt(credit_card=8000, personal_loan=5000)
            investments = Investments(stocks=15000, crypto=2000, bonds=5000, fd=10000,
                                      mutual_funds=8000, commodities=3000, real_estate=0)
            savings = 25000
            emergency_fund = 15000
            credit_score = 700
            market_regime = "bear"   # immediate crash
        else:
            # default balanced
            debt = Debt(credit_card=20000, personal_loan=30000)
            investments = Investments(stocks=5000, crypto=0, bonds=2000, fd=5000,
                                      mutual_funds=3000, commodities=2000, real_estate=0)
            savings = 20000
            emergency_fund = 0
            credit_score = 650
            market_regime = "bull"

        self.state = {
            "month": 1,
            "income": 50000,
            "income_growth": 0.05,
            "fixed_expenses": 20000,
            "variable_expenses": 10000,
            "savings": savings,
            "emergency_fund": emergency_fund,
            "debt": debt,
            "credit_score": credit_score,
            "credit_limit": 50000,
            "credit_used": debt.credit_card,
            "investments": investments,
            "market_regime": market_regime,
            "event": "none"
        }

        obs = Observation(**self.state)
        return obs

    def step(self, action: Action) -> StepResult:
        prev_net = compute_net_worth(self.state)

        # Apply action
        apply_action(self.state, action)
        # Apply interest on debt
        apply_interest(self.state)
        # Market simulation (uses self._rng)
        simulate_market(self.state, self._rng)
        # Rental income
        apply_real_estate_income(self.state)
        # Regime switch
        switch_regime(self.state, self._rng)
        # Random life event
        apply_event(self.state, self._rng)
        # Credit score update
        update_credit_score(self.state)

        self.state["month"] += 1

        curr_net = compute_net_worth(self.state)
        reward = compute_reward(prev_net, curr_net, self.state)

        done = self.state["month"] > self.max_months

        info = Info(
            net_worth=curr_net,
            failures=failure_analysis(self.state),
            regime=self.state["market_regime"],
            event=self.state["event"]
        )

        obs = Observation(**self.state)
        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> dict:
        return self.state.copy()