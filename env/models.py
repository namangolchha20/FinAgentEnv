from pydantic import BaseModel
from typing import List

class Debt(BaseModel):
    credit_card: float
    personal_loan: float

class Investments(BaseModel):
    stocks: float
    crypto: float
    bonds: float
    fd: float
    mutual_funds: float
    commodities: float
    real_estate: float

class Info(BaseModel):
    net_worth: float
    failures: List[str]
    regime: str
    event: str
    cash_flow: float = 0.0          # income - expenses applied this month
    action_ok: bool = True          # whether the submitted action was valid
    action_message: str = ""        # human-readable result of the action

class Observation(BaseModel):
    month: int
    income: float
    income_growth: float
    fixed_expenses: float
    variable_expenses: float
    savings: float
    emergency_fund: float
    debt: Debt
    credit_score: float
    credit_limit: float
    credit_used: float
    investments: Investments
    market_regime: str
    event: str
    net_worth: float = 0.0

ACTION_TYPES = [
    "pay_credit_card",
    "pay_personal_loan",
    "invest_stocks",
    "invest_crypto",
    "invest_bonds",
    "invest_fd",
    "invest_mutual_funds",
    "invest_commodities",
    "sell_stocks",
    "sell_crypto",
    "sell_bonds",
    "sell_fd",
    "sell_mutual_funds",
    "sell_commodities",
    "buy_real_estate",
    "sell_real_estate",
    "build_emergency_fund",
    "withdraw_emergency_fund",
    "reduce_spending",
    "hold",
]

class Action(BaseModel):
    action_type: str   # one of ACTION_TYPES
    amount: float = 0.0

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Info
