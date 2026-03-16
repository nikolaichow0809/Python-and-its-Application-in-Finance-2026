import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Goals-Based Dynamic Programming Demo",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Goals-Based Dynamic Programming Demo")
st.write(
    "Teaching prototype: choose the safest asset that still keeps future goals achievable."
)

# ============================================================
# Helper functions
# ============================================================
def round_to_grid(x: float, step: int, max_wealth: int):
    """Round wealth to the nearest grid point and cap at max_wealth."""
    if x < 0:
        return None
    x = min(x, max_wealth)
    return int(step * round(x / step))


def build_assets(safe_r: float, bal_r: float, growth_r: float):
    """
    Assets are ordered from lowest risk to highest risk.
    The DP will always try Safe first, then Balanced, then Growth.
    """
    return [
        {"name": "Safe", "r": safe_r},
        {"name": "Balanced", "r": bal_r},
        {"name": "Growth", "r": growth_r},
    ]


def solve_dp(schedule_df: pd.DataFrame, assets, max_wealth: int, step: int):
    """
    Backward induction.

    State:
        (t, w)
        t = current period
        w = current wealth on the grid

    Action:
        choose one asset from [Safe, Balanced, Growth]

    Transition:
        w_next = (w + contribution_t - goal_withdrawal_t) * (1 + return_of_asset)

    Objective:
        1. Meet all future goals if possible
        2. Among feasible choices, choose the safest asset
    """
    periods = len(schedule_df)
    wealth_grid = np.arange(0, max_wealth + step, step)

    # feasible[t][w] = can all goals from t onward still be met?
    feasible = [{int(w): False for w in wealth_grid} for _ in range(periods + 1)]
    # policy[t][w] = chosen asset at state (t, w)
    policy = [{int(w): None for w in wealth_grid} for _ in range(periods)]

    # Terminal condition: after the last period, any nonnegative wealth is feasible
    for w in wealth_grid:
        feasible[periods][int(w)] = True

    # Backward induction
    for t in range(periods - 1, -1, -1):
        contrib_t = float(schedule_df.loc[t, "contribution"])
        goal_t = float(schedule_df.loc[t, "goal_withdrawal"])

        for w in wealth_grid:
            w = int(w)

            # Cash available after contribution and withdrawal this period
            cash_after_goal = w + contrib_t - goal_t
            if cash_after_goal < 0:
                feasible[t][w] = False
                policy[t][w] = None
                continue

            chosen_asset = None

            # Try assets from safest to riskiest
            for asset in assets:
                w_next = round_to_grid(
                    cash_after_goal * (1 + asset["r"]),
                    step=step,
                    max_wealth=max_wealth
                )

                if w_next is not None and feasible[t + 1].get(w_next, False):
                    chosen_asset = asset["name"]
                    feasible[t][w] = True
                    policy[t][w] = chosen_asset
                    break

            if chosen_asset is None:
                feasible[t][w] = False
                policy[t][w] = None

    return wealth_grid, feasible, policy


def simulate_policy(
    initial_wealth: float,
    schedule_df: pd.DataFrame,
    assets,
    max_wealth: int,
    step: int,
    feasible,
    policy
):
    """
    Simulate the recommended policy forward from the initial wealth.
    Uses grid wealth for consistency with the DP solution.
    """
    periods = len(schedule_df)
    w = round_to_grid(initial_wealth, step=step, max_wealth=max_wealth)
    start_feasible = feasible[0].get(w, False)

    rows = []

    for t in range(periods):
        contrib_t = float(schedule_df.loc[t, "contribution"])
        goal_t = float(schedule_df.loc[t, "goal_withdrawal"])
        chosen_asset = policy[t].get(w, None)

        wealth_start = w
        cash_after_goal = wealth_start + contrib_t - goal_t

        if chosen_asset is None or cash_after_goal < 0:
            rows.append({
                "period": t,
                "wealth_start": wealth_start,
                "contribution": contrib_t,
                "goal_withdrawal": goal_t,
                "wealth_after_goal": max(cash_after_goal, 0),
                "asset": "No feasible policy",
                "return": np.nan,
                "wealth_next": np.nan
            })
            break

        asset_obj = next(a for a in assets if a["name"] == chosen_asset)
        asset_return = asset_obj["r"]

        w_next = round_to_grid(
            cash_after_goal * (1 + asset_return),
            step=step,
            max_wealth=max_wealth
        )

        rows.append({
            "period": t,
            "wealth_start": wealth_start,
            "contribution": contrib_t,
            "goal_withdrawal": goal_t,
            "wealth_after_goal": cash_after_goal,
            "asset": chosen_asset,
            "return": asset_return,
            "wealth_next": w_next
        })

        w = w_next

    return start_feasible, pd.DataFrame(rows)


def build_frontier(wealth_grid, feasible):
    """
    For each period t, compute the minimum wealth that is still feasible.
    """
    frontier_rows = []
    periods = len(feasible) - 1

    for t in range(periods):
        feasible_ws = [w for w in wealth_grid if feasible[t].get(int(w), False)]
        min_w = min(feasible_ws) if feasible_ws else np.nan
        frontier_rows.append({"period": t, "min_feasible_wealth": min_w})

    return pd.DataFrame(frontier_rows)


# ============================================================
# Plotting functions
# ============================================================
def plot_schedule(schedule_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(schedule_df))
    width = 0.38

    ax.bar(x - width / 2, schedule_df["contribution"], width=width, label="Contribution")
    ax.bar(x + width / 2, schedule_df["goal_withdrawal"], width=width, label="Withdrawal / Goal")

    ax.set_title("Goal Schedule")
    ax.set_xlabel("Period")
    ax.set_ylabel("Amount")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(alpha=0.25)

    return fig


def plot_policy_map(wealth_grid, feasible, policy):
    periods = len(policy)

    asset_to_code = {
        None: -1,
        "Safe": 0,
        "Balanced": 1,
        "Growth": 2
    }

    Z = np.full((len(wealth_grid), periods), -1)

    for t in range(periods):
        for i, w in enumerate(wealth_grid):
            w = int(w)
            if feasible[t].get(w, False):
                Z[i, t] = asset_to_code.get(policy[t].get(w, None), -1)

    cmap = ListedColormap(["lightgray", "#9ecae1", "#fdd0a2", "#fc9272"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm
    )

    ax.set_title("Policy Map: Safest Feasible Asset")
    ax.set_xlabel("Period")
    ax.set_ylabel("Wealth on Grid")
    ax.set_xticks(np.arange(periods))
    ax.set_yticks(np.linspace(0, len(wealth_grid) - 1, 6, dtype=int))
    ax.set_yticklabels([int(wealth_grid[i]) for i in np.linspace(0, len(wealth_grid) - 1, 6, dtype=int)])

    legend_handles = [
        Patch(facecolor="lightgray", edgecolor="black", label="Not feasible"),
        Patch(facecolor="#9ecae1", edgecolor="black", label="Safe"),
        Patch(facecolor="#fdd0a2", edgecolor="black", label="Balanced"),
        Patch(facecolor="#fc9272", edgecolor="black", label="Growth"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    return fig


def plot_frontier(frontier_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        frontier_df["period"],
        frontier_df["min_feasible_wealth"],
        marker="o"
    )

    ax.set_title("Feasibility Frontier")
    ax.set_xlabel("Period")
    ax.set_ylabel("Minimum Wealth Needed")
    ax.grid(alpha=0.25)

    return fig


def plot_wealth_path(path_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))

    if path_df.empty:
        ax.set_title("Recommended Wealth Path")
        ax.text(0.5, 0.5, "No path available", ha="center", va="center")
        return fig

    x_start = list(path_df["period"])
    x_end = [t + 1 for t in path_df["period"]]

    # Wealth at start of each period
    ax.plot(
        x_start,
        path_df["wealth_start"],
        marker="o",
        label="Start of period wealth"
    )

    # Wealth after contribution and withdrawal
    ax.plot(
        x_start,
        path_df["wealth_after_goal"],
        marker="x",
        linestyle="--",
        label="After contribution / withdrawal"
    )

    # Wealth at next period
    ax.plot(
        x_end,
        path_df["wealth_next"],
        marker="s",
        label="Next period wealth"
    )

    for _, row in path_df.iterrows():
        if pd.notna(row["wealth_next"]):
            ax.annotate(
                row["asset"],
                xy=(row["period"] + 0.5, row["wealth_next"]),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8
            )

    ax.set_title("Recommended Wealth Path Under DP Policy")
    ax.set_xlabel("Period")
    ax.set_ylabel("Wealth")
    ax.grid(alpha=0.25)
    ax.legend()

    return fig


# ============================================================
# Default inputs
# ============================================================
default_schedule = pd.DataFrame({
    "period": list(range(6)),
    "contribution": [20, 20, 20, 10, 0, 0],
    "goal_withdrawal": [0, 0, 30, 0, 60, 40]
})

# ============================================================
# Sidebar controls
# ============================================================
st.sidebar.header("Model Inputs")

initial_wealth = st.sidebar.number_input(
    "Initial wealth",
    min_value=0.0,
    value=100.0,
    step=10.0
)

max_wealth = st.sidebar.number_input(
    "Max wealth on grid",
    min_value=100,
    value=300,
    step=50
)

grid_step = st.sidebar.number_input(
    "Wealth grid step",
    min_value=1,
    value=10,
    step=1
)

st.sidebar.markdown("### Asset returns")
safe_r = st.sidebar.number_input("Safe return", value=0.02, step=0.01, format="%.2f")
bal_r = st.sidebar.number_input("Balanced return", value=0.05, step=0.01, format="%.2f")
growth_r = st.sidebar.number_input("Growth return", value=0.08, step=0.01, format="%.2f")

# Basic checks
if not (safe_r <= bal_r <= growth_r):
    st.sidebar.warning("For this teaching demo, use returns ordered as Safe ≤ Balanced ≤ Growth.")

assets = build_assets(safe_r, bal_r, growth_r)

# ============================================================
# Editable schedule
# ============================================================
st.subheader("1) Set contributions and goal withdrawals by period")
st.caption("Edit the table below. A higher-risk asset will only be chosen when safer assets cannot keep future goals feasible.")

schedule_df = st.data_editor(
    default_schedule,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed"
)

# Clean types
schedule_df["period"] = schedule_df["period"].astype(int)
schedule_df["contribution"] = schedule_df["contribution"].astype(float)
schedule_df["goal_withdrawal"] = schedule_df["goal_withdrawal"].astype(float)

# ============================================================
# Solve model
# ============================================================
wealth_grid, feasible, policy = solve_dp(
    schedule_df=schedule_df,
    assets=assets,
    max_wealth=int(max_wealth),
    step=int(grid_step)
)

start_feasible, path_df = simulate_policy(
    initial_wealth=initial_wealth,
    schedule_df=schedule_df,
    assets=assets,
    max_wealth=int(max_wealth),
    step=int(grid_step),
    feasible=feasible,
    policy=policy
)

frontier_df = build_frontier(wealth_grid, feasible)

# Current recommendation
current_state = round_to_grid(initial_wealth, int(grid_step), int(max_wealth))
current_asset = policy[0].get(current_state, None)

# ============================================================
# Top summary
# ============================================================
st.subheader("2) Summary")

c1, c2, c3 = st.columns(3)
c1.metric("Feasible from current initial wealth?", "Yes" if start_feasible else "No")
c2.metric("Recommended asset in period 0", current_asset if current_asset is not None else "No feasible policy")

frontier_now = frontier_df.loc[frontier_df["period"] == 0, "min_feasible_wealth"].iloc[0]
frontier_text = "N/A" if pd.isna(frontier_now) else f"{frontier_now:.0f}"
c3.metric("Minimum feasible wealth at period 0", frontier_text)

st.markdown(
    """
**Decision rule used here**

- Step 1: check whether all remaining goals can still be met  
- Step 2: among feasible choices, choose the **lowest-return / lowest-risk** asset  
- Step 3: move to a higher-return asset only when necessary
"""
)

# ============================================================
# Figures
# ============================================================
st.subheader("3) Figures")

fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    st.pyplot(plot_schedule(schedule_df), use_container_width=True)

with fig_col2:
    st.pyplot(plot_frontier(frontier_df), use_container_width=True)

fig_col3, fig_col4 = st.columns(2)

with fig_col3:
    st.pyplot(plot_policy_map(wealth_grid, feasible, policy), use_container_width=True)

with fig_col4:
    st.pyplot(plot_wealth_path(path_df), use_container_width=True)

# ============================================================
# Tables
# ============================================================
st.subheader("4) Recommended path from the chosen initial wealth")
if path_df.empty:
    st.info("No simulation path available.")
else:
    display_path = path_df.copy()
    display_path["return"] = display_path["return"].apply(lambda x: "" if pd.isna(x) else f"{x:.0%}")
    st.dataframe(display_path, use_container_width=True, hide_index=True)

st.subheader("5) Feasibility frontier table")
st.dataframe(frontier_df, use_container_width=True, hide_index=True)

st.subheader("6) Interpretation")
if start_feasible:
    st.success(
        "This initial wealth is sufficient on the current grid. "
        "The model shows the safest asset choice that still keeps later goals achievable."
    )
else:
    st.error(
        "This initial wealth is not enough to keep all future goals feasible on the current grid. "
        "Try increasing initial wealth, reducing withdrawals, delaying withdrawals, or increasing contributions."
    )

st.caption(
    "Teaching note: this is a deterministic discrete-state dynamic programming prototype. "
    "It is not a production investment engine."
)
