import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Simple Robo-Advisor Demo", page_icon="📈", layout="centered")

# ============================================================
# 1. Teaching questionnaire (paraphrased, classroom use only)
# ============================================================
QUESTION_BANK = {
    "q1": {
        "text": "How would you rate your willingness to take financial risk?",
        "options": [
            "Extremely low", "Very low", "Low", "Average",
            "High", "Very high", "Extremely high"
        ],
        "block": "attitude"
    },
    "q2": {
        "text": "How easily do you adapt when financial plans go wrong?",
        "options": [
            "Very uneasily", "Somewhat uneasily",
            "Somewhat easily", "Very easily"
        ],
        "block": "attitude"
    },
    "q3": {
        "text": "Which word best describes financial risk for you?",
        "options": ["Danger", "Uncertainty", "Opportunity", "Thrill"],
        "block": "attitude"
    },
    "q4": {
        "text": "Which job would you prefer?",
        "options": [
            "Stable income with low uncertainty",
            "Mostly stable income",
            "Not sure",
            "Higher income with some uncertainty",
            "Higher income with substantial uncertainty"
        ],
        "block": "attitude"
    },
    "q5": {
        "text": "In a major financial decision, do you think more about losses or gains?",
        "options": [
            "Always losses", "Usually losses",
            "Usually gains", "Always gains"
        ],
        "block": "attitude"
    },
    "q6": {
        "text": "How much risk have you taken in past financial decisions?",
        "options": ["Very small", "Small", "Medium", "Large", "Very large"],
        "block": "experience"
    },
    "q7": {
        "text": "How much risk are you currently prepared to take?",
        "options": ["Very small", "Small", "Medium", "Large", "Very large"],
        "block": "experience"
    },
    "q8": {
        "text": "How confident are you in making financial decisions?",
        "options": ["None", "A little", "A reasonable amount", "A great deal", "Complete"],
        "block": "experience"
    },
    "q9": {
        "text": "What portfolio decline would start to make you uncomfortable?",
        "options": ["Any fall", "10%", "20%", "33%", "50%", "More than 50%"],
        "block": "loss_tolerance"
    },
    "q10": {
        "text": "Which portfolio mix looks most attractive to you?",
        "options": [
            "All low risk",
            "Mostly low risk",
            "Low risk with some growth",
            "Balanced",
            "Growth-oriented",
            "Mostly high risk",
            "All high risk"
        ],
        "block": "portfolio_preference"
    }
}

BLOCK_WEIGHTS = {
    "attitude": 0.40,
    "experience": 0.25,
    "loss_tolerance": 0.20,
    "portfolio_preference": 0.15
}

# ============================================================
# 2. Scoring helpers
# ============================================================
def answer_to_points(qid, answer_text):
    options = QUESTION_BANK[qid]["options"]
    return options.index(answer_text) + 1

def points_to_0_100(qid, points):
    max_points = len(QUESTION_BANK[qid]["options"])
    return round((points - 1) / (max_points - 1) * 100, 2)

def risk_bucket(score):
    if score < 25:
        return "Very Conservative"
    elif score < 45:
        return "Conservative"
    elif score < 60:
        return "Balanced"
    elif score < 75:
        return "Growth"
    else:
        return "Aggressive"

def model_allocation(bucket):
    table = {
        "Very Conservative": {"Equity": 0.15, "Bond": 0.60, "Gold": 0.10, "Cash": 0.15},
        "Conservative":      {"Equity": 0.30, "Bond": 0.50, "Gold": 0.10, "Cash": 0.10},
        "Balanced":          {"Equity": 0.50, "Bond": 0.35, "Gold": 0.10, "Cash": 0.05},
        "Growth":            {"Equity": 0.70, "Bond": 0.20, "Gold": 0.05, "Cash": 0.05},
        "Aggressive":        {"Equity": 0.85, "Bond": 0.10, "Gold": 0.05, "Cash": 0.00},
    }
    return table[bucket]

def score_response(responses):
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }

    block_values = {k: [] for k in BLOCK_WEIGHTS.keys()}

    for qid, answer_text in responses.items():
        points = answer_to_points(qid, answer_text)
        score100 = points_to_0_100(qid, points)
        block = QUESTION_BANK[qid]["block"]

        record[qid] = answer_text
        record[f"{qid}_points"] = points
        record[f"{qid}_score100"] = score100
        block_values[block].append(score100)

    block_scores = {}
    for block, vals in block_values.items():
        block_scores[block] = round(sum(vals) / len(vals), 2)
        record[f"{block}_score"] = block_scores[block]

    total_score = round(
        sum(block_scores[b] * BLOCK_WEIGHTS[b] for b in BLOCK_WEIGHTS),
        2
    )
    bucket = risk_bucket(total_score)
    allocation = model_allocation(bucket)

    review_flags = []
    if record["q9_score100"] < 20 and record["q10_score100"] > 80:
        review_flags.append("Low drawdown tolerance but aggressive portfolio preference")
    if record["experience_score"] < 35 and total_score > 70:
        review_flags.append("High risk score with low experience/confidence")
    if record["q7_score100"] - record["q6_score100"] > 40:
        review_flags.append("Current stated risk readiness is much higher than past behavior")

    record["risk_score_100"] = total_score
    record["risk_bucket"] = bucket
    record["review_required"] = len(review_flags) > 0
    record["review_notes"] = " | ".join(review_flags) if review_flags else "None"

    return record, allocation, block_scores

# ============================================================
# 3. UI
# ============================================================
st.title("📈 Simple Robo-Advisor Demo")
st.write("This is a classroom prototype: questionnaire → score → bucket → allocation.")

with st.form("risk_form"):
    name = st.text_input("Client name", value="Student A")

    responses = {}
    for qid, q in QUESTION_BANK.items():
        responses[qid] = st.radio(
            q["text"],
            q["options"],
            key=qid
        )

    submitted = st.form_submit_button("Calculate profile")

if submitted:
    result, allocation, block_scores = score_response(responses)

    st.subheader(f"Result for {name}")
    st.metric("Risk score", f"{result['risk_score_100']:.2f}/100")
    st.metric("Risk bucket", result["risk_bucket"])

    st.subheader("Block scores")
    block_df = pd.DataFrame({
        "Block": list(block_scores.keys()),
        "Score": list(block_scores.values())
    })
    st.dataframe(block_df, use_container_width=True, hide_index=True)

    st.subheader("Suggested allocation")
    alloc_df = pd.DataFrame({
        "Asset": list(allocation.keys()),
        "Weight": [f"{v:.0%}" for v in allocation.values()]
    })
    st.dataframe(alloc_df, use_container_width=True, hide_index=True)

    st.subheader("Review flag")
    if result["review_required"]:
        st.warning(result["review_notes"])
    else:
        st.success("No review flag triggered.")

    st.subheader("Raw audit-style output")
    st.json({
        "client_name": name,
        "timestamp": result["timestamp"],
        "risk_score_100": result["risk_score_100"],
        "risk_bucket": result["risk_bucket"],
        "allocation": allocation,
        "review_required": result["review_required"],
        "review_notes": result["review_notes"]
    })
