import asyncio
import os
import sys
import json
from typing import List

# Add current directory to path so 'env' package can be found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from env.environment import FinAgentEnv
from env.models import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

MAX_STEPS = 6
SUCCESS_SCORE_THRESHOLD = 0.6
TASKS = ["debt_trap", "balanced_growth", "adversarial_crash"]

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(f"[END] success={success} steps={steps} score={score:.2f} rewards={rewards}", flush=True)

def get_llm_action(client: OpenAI, obs, step: int, last_reward: float, history: List[str]) -> Action:
    prompt = f"""You are a financial advisor. Based on the current financial state, choose ONE action and an amount (if applicable). 
Current month: {obs.month}
Income: {obs.income}, Fixed expenses: {obs.fixed_expenses}, Variable expenses: {obs.variable_expenses}
Savings: {obs.savings}, Emergency fund: {obs.emergency_fund}
Debt: credit_card={obs.debt.credit_card}, personal_loan={obs.debt.personal_loan}
Credit score: {obs.credit_score}
Investments: stocks={obs.investments.stocks}, crypto={obs.investments.crypto}, bonds={obs.investments.bonds}, real_estate={obs.investments.real_estate}
Market regime: {obs.market_regime}, Event: {obs.event}
Last step reward: {last_reward:.2f}
History: {history[-3:]}

Possible actions: pay_credit_card, pay_personal_loan, invest_stocks, invest_crypto, invest_bonds, invest_fd, invest_mutual_funds, invest_commodities, buy_real_estate, build_emergency_fund, reduce_spending.
Respond with a JSON object: {{"action_type": "...", "amount": <number>}}.
If action does not use amount, set amount to 0."""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        return Action(action_type=data["action_type"], amount=float(data.get("amount", 0)))
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        return Action(action_type="reduce_spending", amount=0)

async def run_task(task_id: str, seed: int = 42):
    env = FinAgentEnv()
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    rewards = []
    history = []

    log_start(task=task_id, env="FinAgentEnv", model=MODEL_NAME)

    try:
        obs = env.reset(task_id=task_id, seed=seed)
    except Exception as e:
        print(f"[DEBUG] reset failed: {e}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return 0.0

    last_reward = 0.0

    for step in range(1, MAX_STEPS + 1):
        try:
            action = get_llm_action(client, obs, step, last_reward, history)
            result = env.step(action)
            obs = result.observation
            reward = result.reward
            done = result.done
            rewards.append(reward)
            last_reward = reward
            history.append(f"Step {step}: {action.action_type} (amount {action.amount}) -> reward {reward:.2f}")
            log_step(step, action.action_type, reward, done, None)
            if done:
                break
        except Exception as e:
            print(f"[DEBUG] step {step} failed: {e}", flush=True)
            log_step(step, "", 0.0, True, str(e))
            break

    # Use the appropriate grader
    try:
        if task_id == "debt_trap":
            from env.graders import grade_debt_trap
            score = grade_debt_trap(env)
        elif task_id == "balanced_growth":
            from env.graders import grade_balanced_growth
            score = grade_balanced_growth(env)
        else:
            from env.graders import grade_adversarial_crash
            score = grade_adversarial_crash(env)
    except Exception as e:
        print(f"[DEBUG] grader failed: {e}", flush=True)
        score = 0.0

    success = score >= SUCCESS_SCORE_THRESHOLD
    log_end(success, len(rewards), score, rewards)
    return score

async def main():
    scores = []
    for task in TASKS:
        score = await run_task(task, seed=42)
        scores.append(score)
    avg = sum(scores) / len(scores)
    print(f"\n=== BASELINE SCORES ===")
    for t, s in zip(TASKS, scores):
        print(f"{t}: {s:.2f}")
    print(f"Average: {avg:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
