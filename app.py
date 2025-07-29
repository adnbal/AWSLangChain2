import streamlit as st
import pandas as pd
import numpy as np
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

# --- Scoring Function ---
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
    scores = np.clip(scores, 0, 100)  # Force valid range
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

    # Debug Scores
    st.write("DEBUG: Score Summary", df_display['Score'].describe())
    st.write("DEBUG: Sample Scores", df_display[['ID', 'Score']].head(10))

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

    # --- Score Distribution ---
    st.markdown("---")
    st.subheader("Credit Score Distribution")
    st.plotly_chart(px.histogram(df_display, x="Score", nbins=20, title="Score Distribution", color_discrete_sequence=['#4CAF50']), use_container_width=True)

    # --- Loan Decision Pie ---
    st.subheader("Loan Decision Breakdown")
    st.plotly_chart(px.pie(df_display, names="Decision", title="Decision Breakdown", color_discrete_sequence=['#66BB6A', '#EF5350']), use_container_width=True)

    # --- Average Feature Values by Decision ---
    st.markdown("---")
    st.subheader("Average Feature Values by Loan Decision (Split by Scale)")

    avg_features = df_display.groupby('Decision')[NUMERICAL_FEATURES].mean().reset_index()
    melted_avg = avg_features.melt(id_vars='Decision', var_name='Credit Feature', value_name='Average Value')

    group_1 = ['DerogCnt', 'CollectCnt', 'BanruptcyInd', 'InqCnt06', 'InqFinanceCnt24',
               'TLCnt03', 'TLCnt12', 'TLCnt24', 'TLCnt', 'TLSatCnt', 'TLDel60Cnt', 'TLBadCnt24',
               'TL75UtilCnt', 'TL50UtilCnt', 'TLDel3060Cnt24', 'TLDel90Cnt24', 'TLDel60CntAll',
               'TLBadDerogCnt', 'TLDel60Cnt24']
    group_2 = ['InqTimeLast', 'TLTimeLast', 'TLBalHCPct', 'TLSatPct', 'TLOpenPct', 'TLOpen24Pct']
    group_3 = ['TLSum', 'TLMaxSum']

    def plot_group(features, title):
        subset = melted_avg[melted_avg['Credit Feature'].isin(features)]
        fig = px.bar(subset, x="Credit Feature", y="Average Value", color="Decision", barmode="group", text_auto=".2f", title=title)
        fig.update_layout(xaxis_tickangle=-45, legend_title="Decision Type", height=500)
        st.plotly_chart(fig, use_container_width=True)

    plot_group(group_1, "Average Counts & Indicators")
    plot_group(group_2, "Average Time & Percentages")
    plot_group(group_3, "Average Monetary Features")

    # --- Improved Top & Bottom Scores ---
    st.markdown("---")
    st.subheader("Top & Bottom Scores Analysis")

    top_scores = df_display.sort_values(by="Score", ascending=False).head(10)
    bottom_scores = df_display.sort_values(by="Score", ascending=True).head(10)

    if top_scores['Score'].sum() == 0 and bottom_scores['Score'].sum() == 0:
        st.warning("Scores are zero or invalid. Please check your input data format.")
    else:
        # Top Scores
        fig_top = px.bar(top_scores.sort_values(by="Score"), x="Score", y="ID", orientation="h",
                         title="Top 10 Highest Credit Scores", text="Score",
                         color_discrete_sequence=["green"])
        fig_top.update_traces(texttemplate='%{text:.2f}', textposition='inside')
        st.plotly_chart(fig_top, use_container_width=True)

        # Bottom Scores
        fig_bottom = px.bar(bottom_scores.sort_values(by="Score"), x="Score", y="ID", orientation="h",
                            title="Top 10 Lowest Credit Scores", text="Score",
                            color_discrete_sequence=["red"])
        fig_bottom.update_traces(texttemplate='%{text:.2f}', textposition='inside')
        st.plotly_chart(fig_bottom, use_container_width=True)

    # --- Summary Statistics ---
    st.markdown("---")
    st.subheader("Summary Statistics")
    summary_stats = df_display['Score'].describe().to_frame().rename(columns={'Score': 'Value'})
    summary_stats.loc['approval_rate'] = approval_rate
    st.dataframe(summary_stats.style.format("{:.2f}"))

    # --- Explanations ---
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
