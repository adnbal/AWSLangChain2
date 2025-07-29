import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import plotly.express as px

# --- Session State ---
st.session_state.setdefault('scored_data', pd.DataFrame())

# --- Page Setup ---
st.set_page_config(page_title="Credit Scoring Dashboard", layout="wide")
st.title("📊 Credit Scoring Dashboard with Adjustable Threshold & Predictors Summary")

# --- Threshold Slider ---
st.markdown("### 🔍 Set Approval Threshold")
approval_threshold = st.slider("Adjust the threshold for loan approval", 0, 100, 50, 1)
st.write(f"Current Approval Threshold: **{approval_threshold}%**")
st.session_state['approval_threshold'] = approval_threshold

# --- Define Features ---
NUMERICAL_FEATURES = [
    'DerogCnt', 'CollectCnt', 'BanruptcyInd', 'InqCnt06', 'InqTimeLast', 'InqFinanceCnt24',
    'TLTimeFirst', 'TLTimeLast', 'TLCnt03', 'TLCnt12', 'TLCnt24', 'TLCnt',
    'TLSum', 'TLMaxSum', 'TLSatCnt', 'TLDel60Cnt', 'TLDel60Cnt24',
    'TLBadCnt24', 'TL75UtilCnt', 'TL50UtilCnt', 'TLBalHCPct', 'TLSatPct', 'TLDel3060Cnt24',
    'TLDel90Cnt24', 'TLDel60CntAll', 'TLOpenPct', 'TLBadDerogCnt', 'TLOpen24Pct'
]

# --- Simulated Realistic Scoring ---
def get_credit_score_realistic(df_input):
    df = df_input.copy()
    # Fill missing required columns with zeros
    missing_cols = [col for col in NUMERICAL_FEATURES if col not in df.columns]
    for col in missing_cols:
        df[col] = 0

    # Simulate realistic credit scores
    np.random.seed(42)
    scores = np.random.normal(75, 10, len(df))  # Mean 75, std dev 10
    scores = np.clip(scores, 0, 100)
    return scores

# --- LLM Explanation ---
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

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload Credit Data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    if 'ID' not in df.columns:
        df['ID'] = df.index.astype(str)

    # Generate scores & decisions
    scores = get_credit_score_realistic(df)
    decisions = np.where(scores >= approval_threshold, "Approved", "Rejected")
    scored_df = pd.DataFrame({'Score': scores, 'Decision': decisions})
    final_df = pd.concat([df, scored_df], axis=1)
    st.session_state['scored_data'] = final_df

# --- Dashboard Display ---
if not st.session_state['scored_data'].empty:
    df_display = st.session_state['scored_data']
    df_display['Score'] = pd.to_numeric(df_display['Score'], errors='coerce').fillna(0)

    # --- Metrics ---
    st.markdown("### 📈 Dashboard Overview")
    col1, col2, col3 = st.columns(3)
    total = len(df_display)
    approved = (df_display['Decision'] == 'Approved').sum()
    approval_rate = approved / total * 100
    col1.metric("Total Loans", total)
    col2.metric("Approved", approved)
    col3.metric("Approval Rate", f"{approval_rate:.2f}%")

    # --- Score Distribution ---
    st.markdown("---")
    st.subheader("📊 Credit Score Distribution")
    st.plotly_chart(px.histogram(df_display, x="Score", nbins=20, title="Distribution of Credit Scores",
                                 color_discrete_sequence=['#4CAF50']), use_container_width=True)

    # --- Loan Decision Pie ---
    st.subheader("📌 Loan Decision Breakdown")
    st.plotly_chart(px.pie(df_display, names="Decision", title="Approved vs Rejected",
                           color_discrete_sequence=['#66BB6A', '#EF5350']), use_container_width=True)

    # --- Top & Bottom Scores ---
    st.markdown("---")
    st.subheader("🏆 Top & Bottom Scores")
    top_scores = df_display.sort_values(by="Score", ascending=False).head(10)
    bottom_scores = df_display.sort_values(by="Score", ascending=True).head(10)

    if not top_scores.empty:
        st.plotly_chart(px.bar(top_scores.sort_values(by="Score"), x="Score", y="ID", orientation="h",
                               title="Top 10 Highest Credit Scores", text="Score",
                               color_discrete_sequence=["green"]), use_container_width=True)

    if not bottom_scores.empty:
        st.plotly_chart(px.bar(bottom_scores.sort_values(by="Score"), x="Score", y="ID", orientation="h",
                               title="Top 10 Lowest Credit Scores", text="Score",
                               color_discrete_sequence=["red"]), use_container_width=True)

    # --- Summary Stats ---
    st.markdown("---")
    st.subheader("📑 Summary Statistics")
    summary_stats = df_display['Score'].describe().to_frame().rename(columns={'Score': 'Value'})
    summary_stats.loc['approval_rate'] = approval_rate
    st.dataframe(summary_stats.style.format("{:.2f}"))

    # ✅ NEW SECTION: Predictor Summary for Approved vs Rejected
    st.markdown("---")
    st.subheader("🔍 Predictor Summary for Approved vs Rejected Loans")

    # Compute average predictor values by Decision
    predictor_summary = df_display.groupby('Decision')[NUMERICAL_FEATURES].mean().T
    predictor_summary = predictor_summary.sort_values(by='Approved', ascending=False)  # Sort by Approved values

    st.write("**Average Feature Values by Loan Status:**")
    st.dataframe(predictor_summary.style.format("{:.2f}"))

    # Visualize top predictors
    top_features = predictor_summary.head(10).reset_index().melt(id_vars='index', var_name='Decision', value_name='Average Value')
    fig_features = px.bar(top_features, x='index', y='Average Value', color='Decision', barmode='group',
                          title="Top 10 Predictors - Avg Value by Loan Status")
    fig_features.update_layout(xaxis_title="Feature", yaxis_title="Avg Value", xaxis_tickangle=-45)
    st.plotly_chart(fig_features, use_container_width=True)

    # --- AI Explanations ---
    st.markdown("---")
    st.subheader("🤖 AI Explanations for Rejected Loans")
    rejected = df_display[df_display['Decision'] == 'Rejected']
    if not rejected.empty:
        for _, row in rejected.iterrows():
            with st.expander(f"Explain Loan ID: {row['ID']} (Score: {row['Score']:.2f})"):
                features_dict = {col: row[col] for col in NUMERICAL_FEATURES if col in df_display.columns}
                explanation = get_llm_explanation(features_dict, row['Score'], row['Decision'])
                st.write(explanation)
    else:
        st.info("✅ No rejected loans at the current threshold.")
else:
    st.info("📤 Upload a CSV or Excel file to start analysis.")
