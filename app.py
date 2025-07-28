import streamlit as st
import pandas as pd
import numpy as np
import boto3
import io
import datetime # For unique file naming
import os
import time # For a small delay

# Import scikit-learn components
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 

# Import Plotly for fancy visualizations
import plotly.express as px
import plotly.graph_objects as go

# --- Initialize session state variables using setdefault for robustness ---
st.session_state.setdefault('scored_data', pd.DataFrame())
st.session_state.setdefault('analyze_triggered', False)
st.session_state.setdefault('last_uploaded_s3_key', None)
st.session_state.setdefault('selected_file_for_analysis', None)
st.session_state.setdefault('file_uploader_key', 0) 
st.session_state.setdefault('approval_threshold', 50) # Default approval threshold

# --- Define Feature Columns for the NEW 'credit.csv' dataset ---
# CHANGED: BanruptcyInd moved to NUMERICAL_FEATURES
NUMERICAL_FEATURES = [
    'DerogCnt', 'CollectCnt', 'BanruptcyInd', 'InqCnt06', 'InqTimeLast', 'InqFinanceCnt24',
    'TLTimeFirst', 'TLTimeLast', 'TLCnt03', 'TLCnt12', 'TLCnt24', 'TLCnt',
    'TLSum', 'TLMaxSum', 'TLSatCnt', 'TLDel60Cnt', 'TLBadCnt24',
    'TL75UtilCnt', 'TL50UtilCnt', 'TLBalHCPct', 'TLSatPct', 'TLDel3060Cnt24',
    'TLDel90Cnt24', 'TLDel60CntAll', 'TLOpenPct', 'TLBadDerogCnt', 'TLOpen24Pct'
]
CATEGORICAL_FEATURES = [] # No explicit categorical features based on the provided header and common usage
TARGET_COLUMN = 'TARGET' # The column to predict

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# --- Simulate Model Training and Preprocessing for the NEW dataset ---
# This dummy data MUST have the same column names and types as your actual CSV.
dummy_data_for_training = pd.DataFrame({
    'DerogCnt': [1, 0, 2, 0, 1, 0],
    'CollectCnt': [0, 1, 0, 0, 0, 1],
    'BanruptcyInd': [0, 1, 0, 0, 0, 1], # Now correctly treated as numerical
    'InqCnt06': [3, 1, 5, 2, 0, 4],
    'InqTimeLast': [10, 5, 15, 8, 20, 12],
    'InqFinanceCnt24': [2, 1, 3, 1, 0, 2],
    'TLTimeFirst': [120, 80, 200, 90, 150, 100],
    'TLTimeLast': [15, 10, 25, 12, 18, 14],
    'TLCnt03': [1, 0, 2, 1, 0, 1],
    'TLCnt12': [3, 2, 5, 3, 1, 4],
    'TLCnt24': [5, 4, 8, 6, 2, 7],
    'TLCnt': [10, 8, 15, 12, 5, 13],
    'TLSum': [50000, 30000, 80000, 45000, 20000, 60000],
    'TLMaxSum': [15000, 10000, 25000, 12000, 8000, 18000],
    'TLSatCnt': [8, 6, 12, 9, 4, 10],
    'TLDel60Cnt': [0, 1, 0, 0, 1, 0],
    'TLBadCnt24': [0, 1, 0, 0, 1, 0],
    'TL75UtilCnt': [2, 1, 3, 1, 0, 2],
    'TL50UtilCnt': [4, 2, 6, 3, 1, 4],
    'TLBalHCPct': [0.6, 0.8, 0.5, 0.7, 0.9, 0.65],
    'TLSatPct': [0.8, 0.7, 0.85, 0.75, 0.6, 0.82],
    'TLDel3060Cnt24': [0, 0, 1, 0, 0, 1],
    'TLDel90Cnt24': [0, 1, 0, 0, 0, 0],
    'TLDel60CntAll': [0, 1, 0, 0, 1, 1],
    'TLOpenPct': [0.5, 0.3, 0.6, 0.4, 0.2, 0.55],
    'TLBadDerogCnt': [0, 1, 0, 0, 1, 0],
    'TLOpen24Pct': [0.7, 0.4, 0.8, 0.5, 0.3, 0.75],
    'TARGET': [0, 1, 0, 0, 1, 1] # Dummy target variable (0: Good, 1: Bad)
})

# Define preprocessing steps
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), 
    ('scaler', StandardScaler())
])

# If there are no categorical features, this transformer is not strictly needed in ColumnTransformer,
# but keeping it for completeness in case you add categorical features later.
# It will simply not be applied if CATEGORICAL_FEATURES is empty.
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing_category')), 
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, NUMERICAL_FEATURES),
        # Only include 'cat' transformer if there are actual categorical features
        *((('cat', categorical_transformer, CATEGORICAL_FEATURES),) if CATEGORICAL_FEATURES else ())
    ])

# Create a pipeline: preprocess then apply HistGradientBoostingClassifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42)) 
])

# "Fit" the pipeline on the dummy data
try:
    model_pipeline.fit(dummy_data_for_training[ALL_FEATURES], dummy_data_for_training[TARGET_COLUMN])
    st.sidebar.success("Simulated ML model (HistGradientBoostingClassifier) and preprocessor initialized for new data.")
except Exception as e:
    st.sidebar.error(f"Error initializing simulated ML model: {e}")
    st.stop() 

# --- Credit Scoring Function using the ML Model ---
@st.cache_data 
def get_credit_score_ml(df_input: pd.DataFrame, approval_threshold: int): 
    """
    Applies the simulated ML model to score a DataFrame of credit applications.
    Performs robust NaN handling and type coercion before passing to the ML pipeline.
    """
    df_processed = df_input.copy()

    # Ensure all expected feature columns exist. Add if missing with default values.
    for col in ALL_FEATURES:
        if col not in df_processed.columns:
            if col in NUMERICAL_FEATURES:
                df_processed[col] = 0.0 
            elif col in CATEGORICAL_FEATURES:
                df_processed[col] = 'missing_category' 
    
    # Coerce numerical columns to numeric type, converting any non-numeric values to NaN
    for col in NUMERICAL_FEATURES:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        # The SimpleImputer in the pipeline will handle these NaNs.
        # No need for explicit fillna here anymore, as HistGradientBoostingClassifier handles NaNs.
        # However, the numerical_transformer's SimpleImputer will still fill them for consistency.

    # Explicitly fill NaNs in categorical columns and ensure they are string type
    for col in CATEGORICAL_FEATURES:
        df_processed[col] = df_processed[col].astype(str).replace('nan', 'missing_category')
        df_processed[col] = df_processed[col].fillna('missing_category') 

    # Select and reorder columns to match the training order
    df_processed_for_prediction = df_processed[ALL_FEATURES]

    st.write("DEBUG: DataFrame info before ML prediction:")
    buffer = io.StringIO()
    df_processed_for_prediction.info(buf=buffer)
    st.text(buffer.getvalue())
    st.write(f"DEBUG: Total NaNs in data before prediction: {df_processed_for_prediction.isnull().sum().sum()}")


    # Predict probabilities (probability of BAD=1, BAD=0)
    # HistGradientBoostingClassifier.predict_proba returns probabilities for each class
    # Assuming class 0 (index 0) is "Good" and class 1 (index 1) is "Bad" based on TARGET column
    probabilities = model_pipeline.predict_proba(df_processed_for_prediction)
    
    scores = probabilities[:, 0] * 100 # Probability of being a "Good" (non-defaulting) applicant (Class 0)

    # Make decisions based on a threshold (can be tuned)
    decisions = np.where(scores >= approval_threshold, "Approved", "Rejected") 

    return pd.DataFrame({'Score': scores, 'Decision': decisions})


# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Credit Scoring Dashboard with S3", layout="wide")

# Custom CSS for styling the Streamlit app
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 8px;
    }
    .stSelectbox>div>div {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 8px;
    }
    .stHeader {
        color: #2c3e50;
        text-align: center;
    }
    .stSubheader {
        color: #34495e;
    }
    .score-box {
        background-color: #e8f5e9;
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
        color: #2e7d32;
    }
    .decision-approved {
        background-color: #e8f5e9;
        border: 2px solid #4CAF50;
        color: #2e7d32;
        padding: 10px;
        border-radius: 8px;
        margin-top: 10px;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
    }
    .decision-rejected {
        background-color: #ffebee;
        border: 2px solid #f44336;
        color: #c62828;
        padding: 10px;
        border-radius: 8px;
        margin-top: 10px;
        font-size: 1.5em;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Credit Risk Dashboard (Powered by Gradient Boosting with Adjustable Threshold!)") 
st.subheader("Upload Credit Data and Analyze Risk") 

# --- Initialize S3 Client ---
s3_client = None
s3_bucket_name = None
aws_region_name = None

try:
    # Accessing secrets as nested keys, matching your Streamlit Cloud config
    aws_access_key = st.secrets["aws"]["access_key_id"]
    aws_secret_key = st.secrets["aws"]["secret_access_key"]
    s3_bucket_name = st.secrets["aws"]["s3_bucket_name"]
    aws_region_name = st.secrets["aws"]["region_name"]

    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region_name
    )
    st.sidebar.success("AWS S3 client initialized!")

    st.sidebar.header("Application Secrets")
    st.sidebar.write(f"AWS Access Key ID: `{aws_access_key[:4]}...`")
    st.sidebar.write(f"AWS Secret Access Key: `{aws_secret_key[:4]}...`")
    st.sidebar.write(f"S3 Bucket Name: `{s3_bucket_name}`")
    st.sidebar.write(f"AWS Region: `{aws_region_name}`")

except KeyError as e:
    st.sidebar.error(f"Secret key not found: {e}. Please ensure your secrets are configured correctly as nested keys under `[aws]` in Streamlit Cloud or `.streamlit/secrets.toml`.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"An error occurred while initializing S3 client: {e}")
    st.stop()

# --- Callback function to clear the file uploader ---
def clear_file_uploader():
    st.session_state['file_uploader_key'] += 1

# --- File Upload Section ---
st.header("Upload Credit Data to S3") 
uploaded_file = st.file_uploader(
    "Choose a CSV file (.csv)", 
    type=["csv"], 
    key=f"file_uploader_{st.session_state['file_uploader_key']}" 
)

if uploaded_file is not None:
    file_name = uploaded_file.name
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    s3_file_key = f"uploads/{timestamp}_{file_name}"

    try:
        with st.spinner(f"Uploading {file_name} to S3..."):
            s3_client.upload_fileobj(uploaded_file, s3_bucket_name, s3_file_key)
        st.success(f"File '{file_name}' uploaded successfully to S3 as '{s3_file_key}'!")
        st.session_state['last_uploaded_s3_key'] = s3_file_key
        st.session_state['analyze_triggered'] = False 
        st.session_state['scored_data'] = pd.DataFrame() 
        st.info("File uploaded. Please select it from the dropdown below and click 'Analyze'.")
        time.sleep(1) 

        clear_file_uploader() 
        st.rerun() 

    except Exception as e:
        st.error(f"Error uploading file to S3: {e}")

# --- Dashboard Section for Analysis Trigger (Encapsulated in a Form) ---
st.header("Analyze Credit Data from S3") 

s3_files = []
try:
    if s3_client:
        response = s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix="uploads/")
        if 'Contents' in response:
            s3_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.csv')] 
            s3_files.sort(reverse=True)
except Exception as e:
    st.warning(f"Could not list files from S3 bucket: {e}. Ensure 'uploads/' prefix exists or bucket is not empty, and permissions are correct.")
    s3_files = []

# --- Use a form to encapsulate the file selection and analysis trigger ---
with st.form("analysis_trigger_form"):
    st.markdown("### Select File for Analysis")
    
    selected_s3_file_in_form = None
    if s3_files:
        default_index = 0
        if st.session_state['last_uploaded_s3_key'] and st.session_state['last_uploaded_s3_key'] in s3_files:
            try:
                default_index = s3_files.index(st.session_state['last_uploaded_s3_key'])
            except ValueError:
                default_index = 0

        selected_s3_file_in_form = st.selectbox(
            "Choose a CSV file from S3:", 
            options=s3_files,
            index=default_index,
            key="s3_file_selector_form" 
        )
    else:
        st.info("No CSV files found in the 'uploads/' folder of your S3 bucket. Please upload one above.")

    # This is the button that submits the form and triggers analysis
    analyze_submitted = st.form_submit_button(f"Analyze Selected File")

    if analyze_submitted:
        if selected_s3_file_in_form:
            st.write(f"DEBUG: Analyze form submitted. Selected file: `{selected_s3_file_in_form}`. Starting analysis...")
            st.session_state['selected_file_for_analysis'] = selected_s3_file_in_form 
            st.session_state['analyze_triggered'] = True 
            st.rerun() 
        else:
            st.warning("Please select a file to analyze.")
            st.session_state['analyze_triggered'] = False
    else:
        st.write("DEBUG: Analysis form not yet submitted.")

# --- Perform Analysis if Triggered ---
if st.session_state['analyze_triggered'] and st.session_state['selected_file_for_analysis']:
    file_to_analyze = st.session_state['selected_file_for_analysis']
    st.write(f"DEBUG: Analysis triggered via session state for file: `{file_to_analyze}`.")
    try:
        with st.spinner(f"Downloading and analyzing '{file_to_analyze}' from S3..."):
            obj = s3_client.get_object(Bucket=s3_bucket_name, Key=file_to_analyze)
            df = pd.read_csv(io.BytesIO(obj['Body'].read())) 
            st.write(f"DEBUG: Successfully read {len(df)} rows from CSV file.")
            st.write("DEBUG: Columns in loaded CSV file:", df.columns.tolist()) 

            # Drop 'ID' column if it exists, as it's not a feature
            if 'ID' in df.columns:
                df = df.drop(columns=['ID'])
                st.write("DEBUG: Dropped 'ID' column.")

            # Ensure TARGET_COLUMN is integer type for classification
            if TARGET_COLUMN in df.columns:
                df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce').fillna(0).astype(int)
                st.write(f"DEBUG: '{TARGET_COLUMN}' column processed to integer type.")
            else:
                st.warning(f"'{TARGET_COLUMN}' column not found in the uploaded data. Please ensure your CSV has a '{TARGET_COLUMN}' column for scoring.")
                st.session_state['scored_data'] = pd.DataFrame()
                st.session_state['analyze_triggered'] = False
                st.session_state['selected_file_for_analysis'] = None
                st.stop() 

            results_df = get_credit_score_ml(df.copy(), st.session_state['approval_threshold']) 
            df_scored = pd.concat([df, results_df], axis=1)
            st.write(f"DEBUG: Scored data has {len(df_scored)} rows.")

            st.session_state['scored_data'] = df_scored
            st.success(f"Analysis complete for '{file_to_analyze}'.")
            st.session_state['analyze_triggered'] = False 
            st.session_state['selected_file_for_analysis'] = None 
            st.write("DEBUG: Analysis complete. Trigger and selected file reset.")

    except Exception as e:
        st.error(f"Error analyzing file from S3: {e}. Please check file format and column names in your CSV file.")
        st.write("DEBUG: An error occurred during analysis:", e)
        st.session_state['scored_data'] = pd.DataFrame()
        st.session_state['analyze_triggered'] = False 
        st.session_state['selected_file_for_analysis'] = None 

# --- Clear Data Button ---
if st.session_state['scored_data'] is not None and not st.session_state['scored_data'].empty:
    if st.button("Clear Displayed Data", key="clear_data_button"):
        st.session_state['scored_data'] = pd.DataFrame()
        st.session_state['analyze_triggered'] = False
        st.session_state['selected_file_for_analysis'] = None
        st.success("Displayed data cleared.")
        st.rerun() 

# --- Display Dashboard Results ---
st.markdown("### Dashboard Display")
if 'scored_data' in st.session_state and not st.session_state['scored_data'].empty:
    df_display = st.session_state['scored_data']
    st.write("DEBUG: Displaying scored data.")

    st.subheader("Full Credit Data with Scores and Decisions") 
    st.dataframe(df_display)

    # Adjustable Approval Threshold Slider
    st.markdown("---")
    st.subheader("Adjust Risk Approval Threshold") 
    st.session_state['approval_threshold'] = st.slider(
        "Set Minimum Score for Approval:",
        min_value=0,
        max_value=100,
        value=st.session_state['approval_threshold'], 
        step=1,
        help="Applicants with a score equal to or above this value will be 'Approved'. Adjust to see impact on decisions.",
        key="approval_threshold_slider"
    )
    st.info(f"Current Approval Threshold: **{st.session_state['approval_threshold']}**")
    st.warning("Changing the threshold will re-evaluate all decisions. Click 'Analyze Selected File' again to apply.")

    # --- Dashboard Visualizations Section ---
    st.header("Credit Performance Dashboard") 
    st.markdown("Dive into the top-performing and most vulnerable credit applications with interactive charts.") 

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Approved Applicants (Highest Scores)") 
        top_10_approved_loans = df_display[df_display['Decision'] == 'Approved'].sort_values(by='Score', ascending=False).head(10)
        
        if not top_10_approved_loans.empty:
            fig_top10 = px.bar(
                top_10_approved_loans,
                x=top_10_approved_loans.index, # Use DataFrame index as x-axis for unique ID representation
                y='Score',
                color_discrete_sequence=['#4CAF50'], 
                title='Top 10 Approved Applicants by Score', 
                labels={top_10_approved_loans.index.name: 'Applicant ID', 'Score': 'Credit Score'}, 
                hover_data=NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ['Decision'] 
            )
            fig_top10.update_layout(xaxis_tickangle=-45) 
            st.plotly_chart(fig_top10, use_container_width=True)
        else:
            st.info("No approved applicants to display in the top 10 chart based on current threshold.")


    with col2:
        st.subheader("Worst 10 Rejected Applicants (Lowest Scores)") 
        worst_10_rejected_loans = df_display[df_display['Decision'] == 'Rejected'].sort_values(by='Score', ascending=True).head(10)
        
        if not worst_10_rejected_loans.empty:
            fig_worst10 = px.bar(
                worst_10_rejected_loans,
                x=worst_10_rejected_loans.index, # Use DataFrame index as x-axis for unique ID representation
                y='Score',
                color_discrete_sequence=['#f44336'], 
                title='Worst 10 Rejected Applicants by Score', 
                labels={worst_10_rejected_loans.index.name: 'Applicant ID', 'Score': 'Credit Score'}, 
                hover_data=NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ['Decision']
            )
            fig_worst10.update_layout(xaxis_tickangle=-45) 
            st.plotly_chart(fig_worst10, use_container_width=True)
        else:
            st.info("No rejected applicants to display in the worst 10 chart based on current threshold.")

    # Overall Score Distribution Histogram
    st.subheader("Overall Credit Score Distribution")
    fig_hist = px.histogram(
        df_display,
        x='Score',
        nbins=20, 
        title='Distribution of Credit Scores',
        labels={'Score': 'Credit Score'},
        color='Decision', 
        color_discrete_map={'Approved': '#4CAF50', 'Rejected': '#f44336'},
        marginal='box' 
    )
    fig_hist.add_vline(x=st.session_state['approval_threshold'], line_width=2, line_dash="dash", line_color="blue", annotation_text=f"Threshold: {st.session_state['approval_threshold']}", annotation_position="top right")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Display the full vulnerable loans table as before
    st.subheader("Rejected Applicants Summary") 
    vulnerable_loans = df_display[df_display['Decision'] == 'Rejected']

    if not vulnerable_loans.empty:
        st.dataframe(vulnerable_loans)
        st.info(f"Found {len(vulnerable_loans)} rejected applicants.") 
    else:
        st.success("No rejected applicants found in this dataset based on current threshold!") 
else:
    st.write("DEBUG: No scored data found in session state or data is empty. Dashboard not displayed.")

# --- AWS and Langchain Integration Explanation (Recap) ---
st.markdown("---")
st.subheader("How this app would integrate with AWS and Langchain (Recap):")

st.markdown(
    """
    This Streamlit app now demonstrates uploading and analyzing **sophisticated credit data** from AWS S3, leveraging a powerful Gradient Boosting Machine learning model.

    ### ☁️ AWS Cloud Integration:
    * **Data Storage (S3):** Data files are stored in AWS S3, providing durable and scalable storage.
    * **Machine Learning Model Hosting (SageMaker):** In a real scenario, the `get_credit_score_ml` function would call a sophisticated ML model deployed on AWS SageMaker for more accurate and robust predictions. This separates the heavy computation from the Streamlit app.
    * **Serverless Functions (Lambda):** Could be used for automated processing of new files uploaded to S3 (e.g., triggering the scoring process automatically).
    * **Authentication & Authorization (Cognito):** For secure user access to the app and S3, ensuring only authorized users can upload or view sensitive data.
    * **Logging & Monitoring (CloudWatch):** To track app performance, S3 interactions, and potential errors, providing insights for operational management.

    ### 🔗 Langchain Integration:
    Langchain is primarily used for building applications with Large Language Models (LLMs). It could enhance this application in several ways:
    * **Explainable AI (XAI):** After identifying rejected applicants, Langchain could prompt an LLM to generate more detailed, human-readable explanations for *why* specific applicants were flagged as high-risk, based on their input features.
    * **Conversational Interface:** Users could interact with the dashboard using natural language queries (e.g., "Show me all applicants with a Derogatory Count greater than 1"), with Langchain interpreting the query and dynamically filtering the DataFrame.
    * **Automated Reporting:** Langchain could help generate summary reports or alerts for high-risk applications, potentially integrating with email services to notify relevant stakeholders.
    """
)
