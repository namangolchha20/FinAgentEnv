import asyncio
import os
import sys
import json
import random
import traceback

# ----------------------------------------------------------------------
# Environment definition (self-contained)
# ----------------------------------------------------------------------

class Debt:
    def __init__(self, credit_card: float, personal_loan: float):
        self.credit_card = credit_card
        self.personal_loan = personal_loan

class Investments:
    def __init__(self, stocks: float, crypto: float, bonds: float, fd: float,
                 mutual_funds: float, commodities: float, real_estate: float):
        self.stocks = stocks
        self.crypto = crypto
        self.bonds = bonds
        self.fd = fd
        self.mutual_funds = mutual_funds
        self.commodities = commodities
        self.real_estate = real_estate

class Observation:
    def __init__(self, month: int, income: float, income_growth: float,
                 fixed_expenses: float, variable_expenses: float,
                 savings: float, emergency_fund: float,
                 debt: Debt, credit_score: float, credit_limit: float,
                 credit_used: float, investments: Investments,
                 market_regime: str, event: str):
        self.month = month
        self.income = income
        self.income_growth = income_growth
        self.fixed_expenses = fixed_expenses
        self.variable_expenses = variable_expenses
        self.savings = savings
        self.emergency_fund = emergency_fund
        self.debt = debt
        self.credit_score = credit_score
        self.credit_limit = credit_limit
        self.credit_used = credit_used
        self.investments = investments
        self.market_regime = market_regime
        self.event = event

class Action:
    def __init__(self, action_type: str, amount: float = 0.0):
        self.action_type = action_type
        self.amount = amount

class Info:
    def __init__(self, net_worth: float, failures: list, regime: str, event: str):
        self.net_worth = net_worth
        self.failures = failures
        self.regime = regime
        self.event = event

class StepResult:
    def __init__(self, observation: Observation, reward: float, done: bool, info: Info):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info

class FinAgentEnv:
    def __init__(self):
        self.max_months = 6
        self._rng = None
        self.state = None

    def reset(self, task_id: str = None, seed: int = None) -> Observation:
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        if task_id == "debt_trap":
            debt = Debt(credit_card=35000, personal_loan=20000)
            investments = Investments(0,0,0,0,0,0,0)
            savings = 5000
            emergency_fund = 0
            credit_score = 580
            market_regime = "sideways"
        elif task_id == "balanced_growth":
            debt = Debt(credit_card=15000, personal_loan=10000)
            investments = Investments(5000,0,2000,5000,3000,2000,0)
            savings = 20000
            emergency_fund = 5000
            credit_score = 650
            market_regime = "bull"
        elif task_id == "adversarial_crash":
            debt = Debt(credit_card=8000, personal_loan=5000)
            investments = Investments(15000,2000,5000,10000,8000,3000,0)
            savings = 25000
            emergency_fund = 15000
            credit_score = 700
            market_regime = "bear"
        else:
            debt = Debt(credit_card=20000, personal_loan=30000)
            investments = Investments(5000,0,2000,5000,3000,2000,0)
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
        return self._make_observation()

    def _make_observation(self):
        return Observation(
            month=self.state["month"],
            income=self.state["income"],
            income_growth=self.state["income_growth"],
            fixed_expenses=self.state["fixed_expenses"],
            variable_expenses=self.state["variable_expenses"],
            savings=self.state["savings"],
            emergency_fund=self.state["emergency_fund"],
            debt=self.state["debt"],
            credit_score=self.state["credit_score"],
            credit_limit=self.state["credit_limit"],
            credit_used=self.state["credit_used"],
            investments=self.state["investments"],
            market_regime=self.state["market_regime"],
            event=self.state["event"]
        )

    def step(self, action: Action) -> StepResult:
        prev_net = self._compute_net_worth()
        self._apply_action(action)
        self._apply_interest()
        self._simulate_market()
        self._apply_real_estate_income()
        self._switch_regime()
        self._apply_event()
        self._update_credit_score()
        self.state["month"] += 1
        curr_net = self._compute_net_worth()
        reward = self._compute_reward(prev_net, curr_net)
        done = self.state["month"] > self.max_months
        info = Info(
            net_worth=curr_net,
            failures=self._failure_analysis(),
            regime=self.state["market_regime"],
            event=self.state["event"]
        )
        return StepResult(self._make_observation(), reward, done, info)

    def _apply_action(self, action):
        amt = action.amount
        if action.action_type == "pay_credit_card":
            payment = min(amt, self.state["savings"])
            self.state["debt"].credit_card -= payment
            self.state["savings"] -= payment
            self.state["credit_used"] -= payment
        elif action.action_type == "pay_personal_loan":
            payment = min(amt, self.state["savings"])
            self.state["debt"].personal_loan -= payment
            self.state["savings"] -= payment
        elif action.action_type == "invest_stocks":
            if self.state["savings"] >= amt:
                self.state["savings"] -= amt
                self.state["investments"].stocks += amt
        elif action.action_type == "invest_crypto":
            if self.state["savings"] >= amt:
                self.state["savings"] -= amt
                self.state["investments"].crypto += amt
        elif action.action_type == "invest_bonds":
            if self.state["savings"] >= amt:
                self.state["savings"] -= amt
                self.state["investments"].bonds += amt
        elif action.action_type == "invest_fd":
            if self.state["savings"] >= amt:
                self.state["savings"] -= amt
                self.state["investments"].fd += amt
        elif action.action_type == "invest_mutual_funds":
            if self.state["savings"] >= amt:
                self.state["savings"] -= amt
                self.state["investments"].mutual_funds += amt
        elif action.action_type == "invest_commodities":
            if self.state["savings"] >= amt:
                self.state["savings"] -= amt
                self.state["investments"].commodities += amt
        elif action.action_type == "buy_real_estate":
            if self.state["savings"] >= 50000:
                self.state["savings"] -= 50000
                self.state["investments"].real_estate += 50000
        elif action.action_type == "build_emergency_fund":
            move = min(amt, self.state["savings"])
            self.state["savings"] -= move
            self.state["emergency_fund"] += move
        elif action.action_type == "reduce_spending":
            self.state["variable_expenses"] *= 0.9

    def _apply_interest(self):
        self.state["debt"].credit_card *= 1.03
        self.state["debt"].personal_loan *= 1.01

    def _simulate_market(self):
        regime = self.state["market_regime"]
        if regime == "bull":
            stock_return = self._rng.uniform(0.05, 0.15)
        elif regime == "bear":
            stock_return = self._rng.uniform(-0.2, -0.05)
        else:
            stock_return = self._rng.uniform(-0.05, 0.05)
        self.state["investments"].stocks *= (1 + stock_return)
        self.state["investments"].crypto *= (1 + self._rng.uniform(-0.3, 0.3))
        self.state["investments"].bonds *= (1 + self._rng.uniform(0.01, 0.05))
        self.state["investments"].fd *= 1.02
        self.state["investments"].mutual_funds *= (1 + stock_return * 0.8)
        self.state["investments"].commodities *= (1 + self._rng.uniform(-0.05, 0.1))
        self.state["investments"].real_estate *= (1 + self._rng.uniform(0.01, 0.05))

    def _apply_real_estate_income(self):
        rental = self.state["investments"].real_estate * 0.01
        self.state["savings"] += rental

    def _switch_regime(self):
        if self._rng.random() < 0.2:
            self.state["market_regime"] = self._rng.choice(["bull", "bear", "sideways"])

    def _apply_event(self):
        events = [("none", 0), ("medical", 10000), ("job_loss", 15000), ("bonus", -15000)]
        event, impact = self._rng.choice(events)
        self.state["event"] = event
        self.state["savings"] = max(0, self.state["savings"] - impact)

    def _update_credit_score(self):
        util = self.state["credit_used"] / self.state["credit_limit"]
        if util > 0.8:
            self.state["credit_score"] -= 20
        elif util < 0.3:
            self.state["credit_score"] += 10
        self.state["credit_score"] = max(300, min(850, self.state["credit_score"]))

    def _compute_net_worth(self) -> float:
        inv = self.state["investments"]
        debt = self.state["debt"]
        return (self.state["savings"] + self.state["emergency_fund"] +
                inv.stocks + inv.crypto + inv.bonds + inv.fd +
                inv.mutual_funds + inv.commodities + inv.real_estate -
                debt.credit_card - debt.personal_loan)

    def _compute_reward(self, prev_net: float, curr_net: float) -> float:
        reward = (curr_net - prev_net) / 20000.0
        total_debt = self.state["debt"].credit_card + self.state["debt"].personal_loan
        reward -= total_debt / 300000.0
        reward += (self.state["credit_score"] - 600) / 1000.0
        if curr_net > prev_net:
            reward += 0.2
        if self.state["emergency_fund"] < 3 * self.state["fixed_expenses"]:
            reward -= 0.3
        else:
            reward += 0.3
        if self.state["savings"] < 5000:
            reward -= 0.2
        return max(-1.0, min(1.0, reward))

    def _failure_analysis(self) -> list:
        failures = []
        if self.state["debt"].credit_card > 20000:
            failures.append("high_credit_card_debt")
        if self.state["emergency_fund"] < 3 * self.state["fixed_expenses"]:
            failures.append("low_emergency_fund")
        if self.state["credit_score"] < 600:
            failures.append("poor_credit_score")
        return failures

    def state(self) -> dict:
        return self.state

# ----------------------------------------------------------------------
# Graders (self-contained) – with strict (0,1) clamping
# ----------------------------------------------------------------------

EPS = 1e-9

def grade_debt_trap(env) -> float:
    state = env.state
    debt_cc = state["debt"].credit_card
    raw = max(0, (35000 - debt_cc) / 35000) * 0.7
    if state["emergency_fund"] >= 10000:
        raw += 0.3
    # Clamp to (0,1)
    return max(EPS, min(1.0 - EPS, raw))

def grade_balanced_growth(env) -> float:
    state = env.state
    net = env._compute_net_worth()
    net_gain = max(0, net - 20000) / 50000
    debt_total = state["debt"].credit_card + state["debt"].personal_loan
    debt_score = max(0, (50000 - debt_total) / 50000) * 0.5
    credit_score = max(0, (state["credit_score"] - 600) / 250) * 0.2
    raw = net_gain * 0.3 + debt_score + credit_score
    return max(EPS, min(1.0 - EPS, raw))

def grade_adversarial_crash(env) -> float:
    state = env.state
    net = env._compute_net_worth()
    net_score = 0.4 if net > 0 else 0.0
    required_ef = 3 * state["fixed_expenses"]
    ef_score = min(1.0, state["emergency_fund"] / required_ef) * 0.4
    credit_score = 0.2 if state["credit_score"] >= 650 else 0.0
    raw = net_score + ef_score + credit_score
    return max(EPS, min(1.0 - EPS, raw))

# ----------------------------------------------------------------------
# Inference script with robust error handling
# ----------------------------------------------------------------------

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    print(f"[END] success={success} steps={steps} score={score:.2f} rewards={rewards}", flush=True)

async def run_task(task_id: str, seed: int = 42):
    # Lazy import of openai inside the function to catch import error
    try:
        from openai import OpenAI
    except ImportError as e:
        print(f"[FATAL] Cannot import openai: {e}", flush=True)
        raise

    env = FinAgentEnv()
    api_base = os.getenv("API_BASE_URL")
    api_key = os.getenv("API_KEY")
    if not api_base or not api_key:
        print("[FATAL] API_BASE_URL or API_KEY environment variables not set", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=api_base, api_key=api_key)
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    max_steps = 6
    success_threshold = 0.6

    rewards = []
    history = []

    log_start(task=task_id, env="FinAgentEnv", model=model_name)

    try:
        obs = env.reset(task_id=task_id, seed=seed)
    except Exception as e:
        print(f"[ERROR] reset failed: {e}", flush=True)
        traceback.print_exc()
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return 0.0

    last_reward = 0.0

    for step in range(1, max_steps + 1):
        try:
            prompt = f"""You are a financial advisor. Based on the current financial state, choose ONE action and an amount.
Current month: {obs.month}
Income: {obs.income}, Fixed expenses: {obs.fixed_expenses}, Variable expenses: {obs.variable_expenses}
Savings: {obs.savings}, Emergency fund: {obs.emergency_fund}
Debt: credit_card={obs.debt.credit_card}, personal_loan={obs.debt.personal_loan}
Credit score: {obs.credit_score}
Investments: stocks={obs.investments.stocks}, crypto={obs.investments.crypto}, bonds={obs.investments.bonds}, real_estate={obs.investments.real_estate}
Market regime: {obs.market_regime}, Event: {obs.event}
Last reward: {last_reward:.2f}
History: {history[-3:]}
Possible actions: pay_credit_card, pay_personal_loan, invest_stocks, invest_crypto, invest_bonds, invest_fd, invest_mutual_funds, invest_commodities, buy_real_estate, build_emergency_fund, reduce_spending.
Respond with JSON: {{"action_type": "...", "amount": <number>}}"""
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            action = Action(data["action_type"], float(data.get("amount", 0)))
        except Exception as e:
            print(f"[ERROR] LLM call failed at step {step}: {e}", flush=True)
            traceback.print_exc()
            # Fallback to a safe action
            action = Action("reduce_spending", 0)

        try:
            result = env.step(action)
            obs = result.observation
            reward = result.reward
            done = result.done
            rewards.append(reward)
            last_reward = reward
            history.append(f"Step {step}: {action.action_type} -> {reward:.2f}")
            log_step(step, action.action_type, reward, done, None)
            if done:
                break
        except Exception as e:
            print(f"[ERROR] step {step} execution failed: {e}", flush=True)
            traceback.print_exc()
            log_step(step, "", 0.0, True, str(e))
            break

    # Grade
    try:
        if task_id == "debt_trap":
            score = grade_debt_trap(env)
        elif task_id == "balanced_growth":
            score = grade_balanced_growth(env)
        else:
            score = grade_adversarial_crash(env)
    except Exception as e:
        print(f"[ERROR] grading failed: {e}", flush=True)
        traceback.print_exc()
        score = 0.0

    success = score >= success_threshold
    log_end(success, len(rewards), score, rewards)
    return score

async def main():
    tasks = ["debt_trap", "balanced_growth", "adversarial_crash"]
    scores = []
    for task in tasks:
        try:
            score = await run_task(task, seed=42)
            scores.append(score)
        except Exception as e:
            print(f"[ERROR] Task {task} failed: {e}", flush=True)
            traceback.print_exc()
            scores.append(0.0)
    avg = sum(scores) / len(scores)
    print(f"\n=== BASELINE SCORES ===")
    for t, s in zip(tasks, scores):
        print(f"{t}: {s:.2f}")
    print(f"Average: {avg:.2f}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"[FATAL] Unhandled exception: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
