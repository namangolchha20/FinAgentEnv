---
title: FinAgentEnv
emoji: 💰
colorFrom: green
colorTo: indigo
sdk: docker
pinned: false
---

# FinAgentEnv – Personal Finance Decision-Making Environment

## Overview
FinAgentEnv simulates realistic personal finance management over 6 months (6 steps). An agent must manage income, expenses, debt, investments, credit score, and unexpected life events (medical, job loss, bonus) under changing market regimes (bull, bear, sideways). The environment is designed for training and evaluating AI agents on real-world financial planning.

## Motivation
Financial decisions involve complex trade-offs: paying debt vs investing, maintaining liquidity vs maximizing returns, managing credit risk vs spending flexibility. This environment provides a structured, reproducible benchmark for these decisions.

## Action Space
The agent chooses one action per step. Actions are typed with an optional amount (float).

| Action Type | Description | Amount usage |
|-------------|-------------|--------------|
| pay_credit_card | Reduce credit card debt | Amount to pay (<= savings) |
| pay_personal_loan | Reduce personal loan debt | Amount to pay (<= savings) |
| invest_stocks | Buy stocks | Amount invested (<= savings) |
| invest_crypto | Buy crypto | Amount invested (<= savings) |
| invest_bonds | Buy bonds | Amount invested (<= savings) |
| invest_fd | Buy fixed deposits | Amount invested (<= savings) |
| invest_mutual_funds | Buy mutual funds | Amount invested (<= savings) |
| invest_commodities | Buy commodities | Amount invested (<= savings) |
| buy_real_estate | Purchase real estate (cost 50,000) | Not used (set 0) |
| build_emergency_fund | Transfer savings to emergency fund | Amount to transfer (<= savings) |
| reduce_spending | Cut variable expenses by 10% | Not used (set 0) |

## Observation Space
Each step returns a structured observation (Pydantic model). Key fields:
- month: current month (1..6)
- income, income_growth, fixed_expenses, variable_expenses
- savings, emergency_fund
- debt: credit_card, personal_loan
- credit_score, credit_limit, credit_used
- investments: stocks, crypto, bonds, fd, mutual_funds, commodities, real_estate
- market_regime: "bull", "bear", or "sideways"
- event: "none", "medical", "job_loss", "bonus"

## Tasks (Easy -> Hard)

| Task ID | Difficulty | Starting Conditions | Goal |
|---------|-----------|---------------------|------|
| debt_trap | Easy | High credit card debt ($35k), low savings ($5k), credit score 580 | Reduce debt, build emergency fund |
| balanced_growth | Medium | Moderate debt ($25k total), savings $20k, credit score 650, bull market | Grow net worth while managing debt and credit |
| adversarial_crash | Hard | Bear market, high investments ($15k stocks), emergency fund $15k, good credit 700 | Survive market downturn and negative events, preserve net worth |

## Reward Function
Dense reward (range approximately -1.0 to 1.0) based on:
- Net worth change (scaled)
- Debt reduction (penalty)
- Credit score improvement (bonus)
- Emergency fund adequacy (bonus/penalty)
- Savings below $5k (penalty)
Partial progress is rewarded at every step.

## Graders
Each task has a deterministic grader returning a score in [0.0, 1.0]:
- debt_trap: focuses on credit card debt reduction and emergency fund.
- balanced_growth: combines net worth growth, debt management, and credit score.
- adversarial_crash: rewards positive net worth, emergency fund size, and credit score >=650.

## Setup & Usage

### Local Installation
pip install -r requirements.txt
pip install -e .

### Validate OpenEnv compliance
openenv validate

### Run Baseline Agent (requires OpenAI API key)
On Windows PowerShell:
$env:OPENAI_API_KEY="your_key_here"
python inference.py

On Linux/macOS:
export OPENAI_API_KEY=your_key_here
python inference.py

### Run Docker Container (server)
docker build -t finagent .
docker run -p 7860:7860 finagent
Then visit http://localhost:7860/docs to interact with the API.

## Baseline Scores (with gpt-4o-mini)
These are estimated; actual scores may vary.

| Task | Score (0-1) |
|------|-------------|
| debt_trap | 0.68 |
| balanced_growth | 0.62 |
| adversarial_crash | 0.55 |
| Average | 0.62 |

## Environment Variables for Inference
- OPENAI_API_KEY (required)
- API_BASE_URL (default https://api.openai.com/v1)
- MODEL_NAME (default gpt-4o-mini)

## OpenEnv Compliance
- Implements reset(task_id, seed) -> Observation, step(action) -> StepResult, state() -> dict
- Uses typed Pydantic models for Observation, Action, StepResult
- Includes openenv.yaml with three tasks and grader references
- Passes openenv validate

## Docker & Hugging Face Space
The environment is packaged as a Docker container listening on port 7860. Deploy to Hugging Face Spaces with Docker SDK. The Space must be public and respond to POST /reset with a valid observation.

## License
MIT