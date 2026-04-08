from pydantic import BaseModel
from typing import List, Optional

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

class Action(BaseModel):
    action_type: str   # one of: pay_credit_card, pay_personal_loan, invest_stocks, invest_crypto, invest_bonds, invest_fd, invest_mutual_funds, invest_commodities, buy_real_estate, build_emergency_fund, reduce_spending
    amount: float = 0.0

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Info