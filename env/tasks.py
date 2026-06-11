"""Task definitions: starting conditions and metadata for each scenario.

Kept in sync with ``openenv.yaml``.
"""

TASKS = {
    "debt_trap": {
        "name": "Debt Trap",
        "difficulty": "easy",
        "description": ("You start drowning in credit card debt with almost no savings. "
                        "Dig out of the hole and build an emergency cushion."),
        "goal": "Reduce credit card debt and build a 10k+ emergency fund.",
        "event_profile": "normal",
        "initial": {
            "debt": {"credit_card": 35000, "personal_loan": 20000},
            "investments": {"stocks": 0, "crypto": 0, "bonds": 0, "fd": 0,
                            "mutual_funds": 0, "commodities": 0, "real_estate": 0},
            "savings": 5000,
            "emergency_fund": 0,
            "credit_score": 580,
            "market_regime": "sideways",
        },
    },
    "balanced_growth": {
        "name": "Balanced Growth",
        "difficulty": "medium",
        "description": ("A healthy start in a bull market. Grow your net worth "
                        "without letting debt or credit slip."),
        "goal": "Grow net worth while paying down debt and protecting your credit score.",
        "event_profile": "normal",
        "initial": {
            "debt": {"credit_card": 15000, "personal_loan": 10000},
            "investments": {"stocks": 5000, "crypto": 0, "bonds": 2000, "fd": 5000,
                            "mutual_funds": 3000, "commodities": 2000, "real_estate": 0},
            "savings": 20000,
            "emergency_fund": 5000,
            "credit_score": 650,
            "market_regime": "bull",
        },
    },
    "adversarial_crash": {
        "name": "Adversarial Crash",
        "difficulty": "hard",
        "description": ("The market just crashed and life keeps throwing punches: "
                        "harsher events, more frequent job losses. Survive."),
        "goal": "Preserve positive net worth, a full emergency fund, and 650+ credit.",
        "event_profile": "adversarial",
        "initial": {
            "debt": {"credit_card": 8000, "personal_loan": 5000},
            "investments": {"stocks": 15000, "crypto": 2000, "bonds": 5000, "fd": 10000,
                            "mutual_funds": 8000, "commodities": 3000, "real_estate": 0},
            "savings": 25000,
            "emergency_fund": 15000,
            "credit_score": 700,
            "market_regime": "bear",
        },
    },
}

DEFAULT_TASK = {
    "name": "Free Play",
    "difficulty": "custom",
    "description": "A balanced default scenario for open-ended experimentation.",
    "goal": "Finish the 6 months in better shape than you started.",
    "event_profile": "normal",
    "initial": {
        "debt": {"credit_card": 20000, "personal_loan": 30000},
        "investments": {"stocks": 5000, "crypto": 0, "bonds": 2000, "fd": 5000,
                        "mutual_funds": 3000, "commodities": 2000, "real_estate": 0},
        "savings": 20000,
        "emergency_fund": 0,
        "credit_score": 650,
        "market_regime": "bull",
    },
}


def get_task(task_id: str | None) -> dict:
    return TASKS.get(task_id, DEFAULT_TASK)
