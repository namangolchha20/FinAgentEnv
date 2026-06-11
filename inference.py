"""Baseline LLM agent for FinAgentEnv.

Imports the canonical environment and graders from the ``env`` package
(no duplicated simulation logic) and runs an OpenAI-compatible model
over all three tasks.

Environment variables:
    OPENAI_API_KEY  (or API_KEY)   required
    API_BASE_URL                   optional, default https://api.openai.com/v1
    MODEL_NAME                     optional, default gpt-4o-mini
"""
import json
import os
import sys
import traceback

from env.environment import FinAgentEnv
from env.models import Action, ACTION_TYPES
from env.graders import grade
from env import config as cfg

SUCCESS_THRESHOLD = 0.6


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done} error={error}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list):
    print(f"[END] success={success} steps={steps} score={score:.2f} rewards={rewards}", flush=True)


def make_client():
    try:
        from openai import OpenAI
    except ImportError as e:
        print(f"[FATAL] Cannot import openai: {e}", flush=True)
        raise

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        print("[FATAL] OPENAI_API_KEY (or API_KEY) environment variable not set", flush=True)
        sys.exit(1)
    api_base = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    return OpenAI(base_url=api_base, api_key=api_key)


def build_prompt(obs, last_reward: float, last_message: str, history: list) -> str:
    inv = obs.investments
    return f"""You are a financial advisor agent. Based on the current financial state, choose ONE action and an amount.
Month: {obs.month}/{cfg.MAX_MONTHS}
Income: {obs.income:.0f}/mo, Fixed expenses: {obs.fixed_expenses:.0f}, Variable expenses: {obs.variable_expenses:.0f}
Savings: {obs.savings:.0f}, Emergency fund: {obs.emergency_fund:.0f} (target: {cfg.EF_TARGET_MONTHS}x fixed expenses)
Debt: credit_card={obs.debt.credit_card:.0f} (3%/mo interest), personal_loan={obs.debt.personal_loan:.0f} (1%/mo interest)
Credit score: {obs.credit_score:.0f} (utilization {obs.credit_used:.0f}/{obs.credit_limit:.0f})
Investments: stocks={inv.stocks:.0f}, crypto={inv.crypto:.0f}, bonds={inv.bonds:.0f}, fd={inv.fd:.0f}, mutual_funds={inv.mutual_funds:.0f}, commodities={inv.commodities:.0f}, real_estate={inv.real_estate:.0f}
Real estate pays {cfg.RENTAL_YIELD_MONTHLY:.0%}/mo rent minus {cfg.MAINTENANCE_RATE_MONTHLY:.1%}/mo maintenance; min purchase {cfg.MIN_PROPERTY_PRICE:.0f}; {cfg.PROPERTY_SALE_FEE:.0%} fee on sale.
Market regime: {obs.market_regime}, Last event: {obs.event}
Net worth: {obs.net_worth:.0f}
Last reward: {last_reward:.2f}. Last action result: {last_message or 'n/a'}
History: {history[-3:]}
Possible actions: {", ".join(ACTION_TYPES)}.
Invalid or unaffordable actions are penalized. Respond with JSON: {{"action_type": "...", "amount": <number>}}"""


def run_task(client, model_name: str, task_id: str, seed: int = 42) -> float:
    env = FinAgentEnv()
    rewards, history = [], []
    log_start(task=task_id, env="FinAgentEnv", model=model_name)

    try:
        obs = env.reset(task_id=task_id, seed=seed)
    except Exception as e:
        print(f"[ERROR] reset failed: {e}", flush=True)
        traceback.print_exc()
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return 0.0

    last_reward = 0.0
    last_message = ""

    for step_n in range(1, cfg.MAX_MONTHS + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user",
                           "content": build_prompt(obs, last_reward, last_message, history)}],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            action = Action(action_type=data["action_type"],
                            amount=float(data.get("amount", 0)))
        except Exception as e:
            print(f"[ERROR] LLM call failed at step {step_n}: {e}", flush=True)
            traceback.print_exc()
            action = Action(action_type="hold", amount=0)

        try:
            result = env.step(action)
            obs = result.observation
            last_reward = result.reward
            last_message = result.info.action_message
            rewards.append(result.reward)
            history.append(f"Step {step_n}: {action.action_type} -> {result.reward:.2f}")
            log_step(step_n, action.action_type, result.reward, result.done)
            if result.done:
                break
        except Exception as e:
            print(f"[ERROR] step {step_n} execution failed: {e}", flush=True)
            traceback.print_exc()
            log_step(step_n, "", 0.0, True, str(e))
            break

    try:
        score = grade(task_id, env)
    except Exception as e:
        print(f"[ERROR] grading failed: {e}", flush=True)
        traceback.print_exc()
        score = 0.0

    success = score >= SUCCESS_THRESHOLD
    log_end(success, len(rewards), score, [round(r, 2) for r in rewards])
    return score


def main():
    client = make_client()
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

    tasks = ["debt_trap", "balanced_growth", "adversarial_crash"]
    scores = []
    for task in tasks:
        try:
            scores.append(run_task(client, model_name, task, seed=42))
        except Exception as e:
            print(f"[ERROR] Task {task} failed: {e}", flush=True)
            traceback.print_exc()
            scores.append(0.0)

    print("\n=== BASELINE SCORES ===")
    for t, s in zip(tasks, scores):
        print(f"{t}: {s:.2f}")
    print(f"Average: {sum(scores) / len(scores):.2f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] Unhandled exception: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
