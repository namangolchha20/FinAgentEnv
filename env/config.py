"""Central configuration for the FinAgentEnv simulation.

All tunable constants live here so tasks, graders, and the engine
share a single source of truth.
"""

# ---------------------------------------------------------------- episode
MAX_MONTHS = 6

# ---------------------------------------------------------------- household
STARTING_INCOME = 50_000.0
INCOME_GROWTH_ANNUAL = 0.05          # applied monthly as /12
FIXED_EXPENSES = 20_000.0
VARIABLE_EXPENSES = 10_000.0
VARIABLE_EXPENSES_FLOOR = 4_000.0    # reduce_spending can't cut below this

# ---------------------------------------------------------------- credit
CREDIT_LIMIT = 50_000.0
CC_MONTHLY_RATE = 0.03
LOAN_MONTHLY_RATE = 0.01
CREDIT_SCORE_MIN = 300
CREDIT_SCORE_MAX = 850

# ---------------------------------------------------------------- real estate
MIN_PROPERTY_PRICE = 50_000.0
RENTAL_YIELD_MONTHLY = 0.01          # 1% of property value paid as rent
MAINTENANCE_RATE_MONTHLY = 0.002     # 0.2% of property value
PROPERTY_SALE_FEE = 0.05             # 5% transaction cost on sale

# ---------------------------------------------------------------- markets
REGIME_SWITCH_PROB = 0.2
REGIMES = ("bull", "bear", "sideways")

# monthly return ranges (low, high) per asset, per regime
MARKET_RETURNS = {
    "bull": {
        "stocks": (0.05, 0.15),
        "crypto": (-0.15, 0.40),
        "bonds": (0.005, 0.03),
        "commodities": (-0.05, 0.08),
        "real_estate": (0.02, 0.06),
    },
    "bear": {
        "stocks": (-0.20, -0.05),
        "crypto": (-0.40, 0.10),
        "bonds": (0.02, 0.06),       # flight to safety
        "commodities": (0.00, 0.12),
        "real_estate": (-0.04, 0.01),
    },
    "sideways": {
        "stocks": (-0.05, 0.05),
        "crypto": (-0.30, 0.30),
        "bonds": (0.01, 0.04),
        "commodities": (-0.05, 0.10),
        "real_estate": (0.00, 0.03),
    },
}
FD_MONTHLY_RATE = 0.02               # fixed deposits: guaranteed return
MUTUAL_FUND_BETA = 0.8               # mutual funds track stocks at 0.8x

# ---------------------------------------------------------------- life events
# (name, weight, kind, low, high)
#   kind "cost"   -> savings hit of rng.uniform(low, high)
#   kind "gain"   -> savings boost of rng.uniform(low, high)
#   kind "income" -> next month's salary is skipped (job_loss)
#   kind "raise"  -> permanent salary raise of rng.uniform(low, high) fraction
EVENT_TABLES = {
    "normal": [
        ("none",         45, "none",   0, 0),
        ("medical",      10, "cost",   8_000, 15_000),
        ("car_repair",   10, "cost",   3_000, 8_000),
        ("home_repair",   8, "cost",   4_000, 10_000),
        ("job_loss",      5, "income", 0, 0),
        ("bonus",        12, "gain",   10_000, 20_000),
        ("tax_refund",    7, "gain",   3_000, 8_000),
        ("salary_raise",  3, "raise",  0.03, 0.08),
    ],
    "adversarial": [
        ("none",         25, "none",   0, 0),
        ("medical",      18, "cost",   10_000, 18_000),
        ("car_repair",   12, "cost",   4_000, 9_000),
        ("home_repair",  12, "cost",   5_000, 12_000),
        ("job_loss",     15, "income", 0, 0),
        ("bonus",         8, "gain",   8_000, 15_000),
        ("tax_refund",    7, "gain",   2_000, 6_000),
        ("salary_raise",  3, "raise",  0.02, 0.05),
    ],
}

# ---------------------------------------------------------------- reward
REWARD_NET_WORTH_SCALE = 20_000.0
REWARD_DEBT_SCALE = 300_000.0
REWARD_CREDIT_SCALE = 1_000.0
REWARD_GROWTH_BONUS = 0.2
REWARD_EF_BONUS = 0.3
REWARD_LOW_SAVINGS_PENALTY = 0.2
REWARD_INVALID_ACTION_PENALTY = 0.1
LOW_SAVINGS_THRESHOLD = 5_000.0
EF_TARGET_MONTHS = 3                 # emergency fund target = 3x fixed expenses
