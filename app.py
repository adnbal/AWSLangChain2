import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import plotly.express as px

# --- Session State ---
st.session_state.setdefault('scored_data', pd.DataFrame())
st.session_state.setdefault('approval_threshold', 50)  # Default threshold

# --- Define Features ---
NUMERICAL_FEATURES = [
    'DerogCnt', 'CollectCnt', 'BanruptcyInd', 'InqCnt06', 'InqTimeLast', 'InqFinanceCnt24',
    'TLTimeFirst', 'TLTimeLast', 'TLCnt03', 'TLCnt12', 'TLCnt24', 'TLCnt',
    'TLSum', 'TLMaxSum', 'TLSatCnt', 'TLDel60Cnt', 'TLDel60Cnt24',
    'TLBadCnt24', 'TL75UtilCnt', 'TL50UtilCnt', 'TLBalHCPct', 'TLSatPct', 'TLDel3060Cnt24',
    'TLDel90Cnt24', 'TLDel60CntAll', 'TLOpenPct', 'TLBadDerogCnt', 'TLOpen24Pct'
]

# --- Simulated Realistic Scoring ---
def get_credit_score_realistic(df_input, approval_threshold):
    df = df_input.copy()

    # Fill missing required columns with zeros
    missing_cols = [col for col in NUMERICAL_FEATURES if col not in df.columns]
    for col in missing_cols:
        df[col] = 0

    # Simulate realistic credit scores (mean ~75, std dev 10)
    np.random.seed(42)  # Reproducibility
    scores = np.random.normal(75, 10, len(df))  # Mean 75, std dev 10
    scores = np.clip(scores, 0, 100)  # Keep between 0 and 100

    # Apply threshold for approval/rejection
    decisions = np.where(scores >= approval_threshold, "Approved", "Rejected")
    return pd.DataFrame({'Score': scores, 'Decision': decisions})

# --- LLM Explanation (Optional) ---
def get_llm_explanation(features_dict, score, decision):
    prompt = f"""
    Explain briefly why this loan has score {score:.2f} and was {decision}.
    Features:
    {json.dumps(features_dict, indent=2)}
    """
    try:
        api_key = st.secrets["openai"]["api_key"]
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100
            }
        )
        return response.json()['choices'][0]['message']['content']
    except:
        return "Explanation unavailable."

# --- Streamlit Layout ---
st.set_page_config(page_title="Credit Scoring Dashboard", layout="wide")
st.title("Credit Scoring Dashboard (Realistic Scores)")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload Credit Data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    # Ensure ID column exists
    if 'ID' not in df.columns:
        df['ID'] = df.index.astype(str)

    # Generate realistic scores
    scored_df = get_credit_score_realistic(df, st.session_state['approval_threshold'])
    final_df = pd.concat([df, scored_df], axis=1)
    st.session_state['scored_data'] = final_df

# --- Dashboard Display ---
if not st.session_state['scored_data'].empty:
    df_display = st.session_state['scored_data']
    df_display['Score'] = pd.to_numeric(df_display['Score'], errors='coerce').fillna(0)

    # Debug Info
    st.write("DEBUG: Score Summary", df_display['Score'].describe())

    # Metrics
    st.subheader("Dashboard")
    st.dataframe(df_display.head())

    col1, col2, col3 = st.columns(3)
    total = len(df_display)
    approved = (df_display['Decision'] == 'Approved').sum()
    approval_rate = approved / total * 100
    col1.metric("Total Loans", total)
    col2.metric("Approved", approved)
    col3.metric("Approval Rate", f"{approval_rate:.2f}%")

    # --- Score Distribution ---
    st.markdown("---")
    st.subheader("Credit Score Distribution")
    fig_hist = px.histogram(df_display, x="Score", nbins=20, title="Distribution of Credit Scores", color_discrete_sequence=['#4CAF50'])
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- Decision Pie Chart ---
    st.subheader("Loan Decision Breakdown")
    fig_pie = px.pie(df_display, names="Decision", title="Proportion of Approved vs Rejected", color_discrete_sequence=['#66BB6A', '#EF5350'])
    st.plotly_chart(fig_pie, use_container_width=True)

    # --- Top & Bottom Scores ---
    st.markdown("---")
    st.subheader("Top & Bottom Scores Analysis")

    top_scores = df_display.sort_values(by="Score", ascending=False).head(10)
    bottom_scores = df_display.sort_values(by="Score", ascending=True).head(10)

    if top_scores['Score'].sum() == 0 or bottom_scores['Score'].sum() == 0:
        st.warning("⚠ No valid score variation found. Check your input data or scoring function.")
    else:
        # Top Scores (Green)
        fig_top = px.bar(top_scores.sort_values(by="Score"), x="Score", y="ID",
                         orientation="h", title="Top 10 Highest Credit Scores",
                         text="Score", color_discrete_sequence=["green"])
        fig_top.update_traces(texttemplate='%{text:.2f}', textposition='inside')
        st.plotly_chart(fig_top, use_container_width=True)

        # Bottom Scores (Red)
        fig_bottom = px.bar(bottom_scores.sort_values(by="Score"), x="Score", y="ID",
                            orientation="h", title="Top 10 Lowest Credit Scores",
                            text="Score", color_discrete_sequence=["red"])
        fig_bottom.update_traces(texttemplate='%{text:.2f}', textposition='inside')
        st.plotly_chart(fig_bottom, use_container_width=True)

    # --- Summary Stats ---
    st.markdown("---")
    st.subheader("Summary Statistics for Credit Scores")
    summary_stats = df_display['Score'].describe().to_frame().rename(columns={'Score': 'Value'})
    summary_stats.loc['approval_rate'] = approval_rate
    st.dataframe(summary_stats.style.format("{:.2f}"))

    # --- AI Explanations for Rejected Loans ---
    st.markdown("---")
    st.subheader("AI Explanations for Rejected Loans")
    rejected = df_display[df_display['Decision'] == 'Rejected']
    if not rejected.empty:
        for _, row in rejected.iterrows():
            with st.expander(f"Explain Loan ID: {row['ID']} (Score: {row['Score']:.2f})"):
                features_dict = {col: row[col] for col in NUMERICAL_FEATURES if col in df_display.columns}
                explanation = get_llm_explanation(features_dict, row['Score'], row['Decision'])
                st.write(explanation)
    else:
        st.info("No rejected loans at the current threshold.")
else:
    st.info("Upload a CSV or Excel file to start analysis.")
