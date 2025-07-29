import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import requests
import random
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --- Session State ---
st.session_state.setdefault('scored_data', pd.DataFrame())
st.session_state.setdefault('approval_threshold', 50)

# --- Define Features ---
NUMERICAL_FEATURES = [
    'DerogCnt', 'CollectCnt', 'BanruptcyInd', 'InqCnt06', 'InqTimeLast', 'InqFinanceCnt24',
    'TLTimeFirst', 'TLTimeLast', 'TLCnt03', 'TLCnt12', 'TLCnt24', 'TLCnt',
    'TLSum', 'TLMaxSum', 'TLSatCnt', 'TLDel60Cnt', 'TLDel60Cnt24',
    'TLBadCnt24', 'TL75UtilCnt', 'TL50UtilCnt', 'TLBalHCPct', 'TLSatPct', 'TLDel3060Cnt24',
    'TLDel90Cnt24', 'TLDel60CntAll', 'TLOpenPct', 'TLBadDerogCnt', 'TLOpen24Pct'
]

# Dummy training data
DUMMY_TARGET_COLUMN_NAME = 'TARGET'
dummy_data_for_training = pd.concat([
    pd.DataFrame({
        'ID': [f'Good_{i}' for i in range(20)],
        **{f: random.randint(0, 5) for f in NUMERICAL_FEATURES},
        DUMMY_TARGET_COLUMN_NAME: [0]*20
    }),
    pd.DataFrame({
        'ID': [f'Bad_{i}' for i in range(10)],
        **{f: random.randint(3, 8) for f in NUMERICAL_FEATURES},
        DUMMY_TARGET_COLUMN_NAME: [1]*10
    })
], ignore_index=True)

# --- Initialize Model ---
@st.cache_resource
def get_model_pipeline():
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[('num', numerical_transformer, NUMERICAL_FEATURES)]
    )
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
    ])
    model_pipeline.fit(dummy_data_for_training[NUMERICAL_FEATURES], dummy_data_for_training[DUMMY_TARGET_COLUMN_NAME])
    return model_pipeline

model_pipeline = get_model_pipeline()

# --- Score Function ---
def get_credit_score_ml(df_input, approval_threshold):
    df = df_input.copy()
    missing_cols = [f for f in NUMERICAL_FEATURES if f not in df.columns]
    if missing_cols:
        st.warning(f"Missing columns filled with 0: {missing_cols}")
        for col in missing_cols:
            df[col] = 0.0
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    probabilities = model_pipeline.predict_proba(df[NUMERICAL_FEATURES])
    scores = probabilities[:, 0] * 100
    decisions = np.where(scores >= approval_threshold, "Approved", "Rejected")
    return pd.DataFrame({'Score': scores, 'Decision': decisions})

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

# --- Streamlit Layout ---
st.set_page_config(page_title="Credit Scoring Dashboard", layout="wide")
st.title("Credit Scoring Dashboard")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload Credit Data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    if 'ID' not in df.columns:
        df['ID'] = df.index.astype(str)
    scored_df = get_credit_score_ml(df, st.session_state['approval_threshold'])
    final_df = pd.concat([df, scored_df], axis=1)
    st.session_state['scored_data'] = final_df

if not st.session_state['scored_data'].empty:
    df_display = st.session_state['scored_data']
    st.subheader("Dashboard")
    st.dataframe(df_display.head())

    # Metrics
    col1, col2, col3 = st.columns(3)
    total = len(df_display)
    approved = (df_display['Decision'] == 'Approved').sum()
    approval_rate = approved / total * 100
    col1.metric("Total", total)
    col2.metric("Approved", approved)
    col3.metric("Approval Rate", f"{approval_rate:.2f}%")

    # --- Charts ---
    st.markdown("---")
    st.subheader("Credit Score Distribution")
    st.plotly_chart(px.histogram(df_display, x="Score", nbins=20, title="Score Distribution"), use_container_width=True)

    st.subheader("Loan Decision Breakdown")
    st.plotly_chart(px.pie(df_display, names="Decision", title="Decision Breakdown"), use_container_width=True)

    # --- Top & Bottom Scores (Ordered Horizontal) ---
    st.markdown("---")
    st.subheader("Top & Bottom Scores Analysis")

    # Top 10 Highest Scores
    top_scores = df_display.sort_values(by="Score", ascending=False).head(10).sort_values(by="Score", ascending=True)
    fig_top = px.bar(
        top_scores,
        x="Score", y="ID",
        orientation="h",
        title="Top 10 Highest Credit Scores",
        color="Score",
        text="Score"
    )
    fig_top.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig_top, use_container_width=True)

    # Top 10 Lowest Scores
    bottom_scores = df_display.sort_values(by="Score", ascending=True).head(10)
    fig_bottom = px.bar(
        bottom_scores.sort_values(by="Score", ascending=False),
        x="Score", y="ID",
        orientation="h",
        title="Top 10 Lowest Credit Scores",
        color="Score",
        text="Score"
    )
    fig_bottom.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig_bottom, use_container_width=True)

    # --- Summary Statistics ---
    st.markdown("---")
    st.subheader("Summary Statistics for Credit Scores")
    summary_stats = df_display['Score'].describe().to_frame().rename(columns={'Score': 'Value'})
    summary_stats.loc['approval_rate'] = approval_rate
    st.dataframe(summary_stats.style.format("{:.2f}"))

    # --- Explanations for Rejected Loans ---
    st.markdown("---")
    st.subheader("AI Explanations for Rejected Loans")
    rejected = df_display[df_display['Decision'] == 'Rejected']
    if not rejected.empty:
        for _, row in rejected.iterrows():
            with st.expander(f"Explain Loan ID: {row['ID']} (Score: {row['Score']:.2f})"):
                explanation = get_llm_explanation(row[NUMERICAL_FEATURES].to_dict(), row['Score'], row['Decision'])
                st.write(explanation)
    else:
        st.info("No rejected loans at the current threshold.")
else:
    st.info("Upload a CSV or Excel file to begin analysis.")
