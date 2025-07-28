import streamlit as st
import pandas as pd
import numpy as np
import boto3
import io
import datetime # For unique file naming
import os
import time # For a small delay
import json # For parsing LLM structured output
import asyncio # For running async functions in a synchronous context

# Import scikit-learn components
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier 
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
st.session_state.setdefault('uploaded_df_columns', []) # To store columns of the last uploaded file
st.session_state.setdefault('selected_target_column', 'default') # Default target column name is 'default' (lowercase)

# --- Define Feature Columns for the 'credit.csv' dataset ---
NUMERICAL_FEATURES = [
    'DerogCnt', 'CollectCnt', 'BanruptcyInd', 'InqCnt06', 'InqTimeLast', 'InqFinanceCnt24',
    'TLTimeFirst', 'TLTimeLast', 'TLCnt03', 'TLCnt12', 'TLCnt24', 'TLCnt',
    'TLSum', 'TLMaxSum', 'TLSatCnt', 'TLDel60Cnt', 'TLBadCnt24',
    'TL75UtilCnt', 'TL50UtilCnt', 'TLBalHCPct', 'TLSatPct', 'TLDel3060Cnt24',
    'TLDel90Cnt24', 'TLDel60CntAll', 'TLOpenPct', 'TLBadDerogCnt', 'TLOpen24Pct'
]
CATEGORICAL_FEATURES = [] # No explicit categorical features based on previous analysis

# Target column name is 'default' (lowercase)
TARGET_COLUMN = 'default' 
DUMMY_TARGET_COLUMN_NAME = TARGET_COLUMN 

# --- Simulate Model Training and Preprocessing for the NEW dataset ---
# This dummy data is significantly expanded and designed to force the model
# to learn patterns for both defaulting (1) and non-defaulting (0) cases.
# It includes more varied values and clearer distinctions.
# Added 'ID' column to dummy data for consistency with actual data structure
dummy_data_for_training = pd.concat([
    pd.DataFrame({
        'ID': [f'Good_ID_{i}' for i in range(20)], # Add dummy IDs
        'DerogCnt': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'CollectCnt': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'BanruptcyInd': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'InqCnt06': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'InqTimeLast': [100, 120, 110, 130, 90, 115, 125, 105, 95, 135, 100, 120, 110, 130, 90, 115, 125, 105, 95, 135],
        'InqFinanceCnt24': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'TLTimeFirst': [200, 250, 220, 280, 180, 230, 260, 210, 190, 290, 200, 250, 220, 280, 180, 230, 260, 210, 190, 290],
        'TLTimeLast': [30, 40, 35, 45, 25, 38, 42, 32, 28, 48, 30, 40, 35, 45, 25, 38, 42, 32, 28, 48],
        'TLCnt03': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'TLCnt12': [3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2],
        'TLCnt24': [5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4],
        'TLCnt': [10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8],
        'TLSum': [50000, 60000, 55000, 65000, 45000, 58000, 62000, 52000, 48000, 68000, 50000, 60000, 55000, 65000, 45000, 58000, 62000, 52000, 48000, 68000],
        'TLMaxSum': [15000, 18000, 16000, 20000, 12000, 17000, 19000, 14000, 13000, 22000, 15000, 18000, 16000, 20000, 12000, 17000, 19000, 14000, 13000, 22000],
        'TLSatCnt': [8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9],
        'TLDel60Cnt': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'TLBadCnt24': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'TL75UtilCnt': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'TL50UtilCnt': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'TLBalHCPct': [0.2, 0.15, 0.25, 0.18, 0.3, 0.22, 0.19, 0.28, 0.21, 0.17, 0.2, 0.15, 0.25, 0.18, 0.3, 0.22, 0.19, 0.28, 0.21, 0.17],
        'TLSatPct': [0.95, 0.98, 0.96, 0.97, 0.94, 0.97, 0.95, 0.98, 0.96, 0.97, 0.95, 0.98, 0.96, 0.97, 0.94, 0.97, 0.95, 0.98, 0.96, 0.97],
        'TLDel3060Cnt24': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'TLDel90Cnt24': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'TLDel60CntAll': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'TLOpenPct': [0.8, 0.75, 0.82, 0.78, 0.85, 0.79, 0.81, 0.77, 0.83, 0.76, 0.8, 0.75, 0.82, 0.78, 0.85, 0.79, 0.81, 0.77, 0.83, 0.76],
        'TLBadDerogCnt': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'TLOpen24Pct': [0.9, 0.88, 0.92, 0.89, 0.95, 0.91, 0.93, 0.87, 0.94, 0.90, 0.9, 0.88, 0.92, 0.89, 0.95, 0.91, 0.93, 0.87, 0.94, 0.90],
        DUMMY_TARGET_COLUMN_NAME: [0] * 20
    }),
    pd.DataFrame({ # Bad applicants (Default = 1) - high derogatory, low credit age, high inquiries
        'ID': [f'Bad_ID_{i}' for i in range(10)], # Add dummy IDs
        'DerogCnt': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        'CollectCnt': [1, 1, 2, 1, 1, 2, 1, 1, 2, 1],
        'BanruptcyInd': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'InqCnt06': [5, 4, 6, 5, 4, 6, 5, 4, 6, 5],
        'InqTimeLast': [10, 20, 15, 25, 12, 18, 22, 14, 28, 16],
        'InqFinanceCnt24': [2, 3, 4, 2, 3, 4, 2, 3, 4, 2],
        'TLTimeFirst': [50, 40, 60, 35, 70, 45, 55, 30, 65, 38],
        'TLTimeLast': [5, 8, 6, 9, 4, 7, 10, 3, 11, 5],
        'TLCnt03': [2, 3, 4, 2, 3, 4, 2, 3, 4, 2],
        'TLCnt12': [6, 7, 8, 6, 7, 8, 6, 7, 8, 6],
        'TLCnt24': [10, 12, 15, 10, 12, 15, 10, 12, 15, 10],
        'TLCnt': [15, 18, 20, 15, 18, 20, 15, 18, 20, 15],
        'TLSum': [20000, 25000, 18000, 30000, 22000, 28000, 21000, 26000, 23000, 29000],
        'TLMaxSum': [5000, 7000, 4000, 8000, 6000, 7500, 5500, 6500, 4500, 8500],
        'TLSatCnt': [2, 3, 1, 4, 2, 3, 1, 4, 2, 3],
        'TLDel60Cnt': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        'TLBadCnt24': [1, 1, 2, 1, 1, 2, 1, 1, 2, 1],
        'TL75UtilCnt': [3, 4, 2, 5, 3, 4, 2, 5, 3, 4],
        'TL50UtilCnt': [5, 6, 4, 7, 5, 6, 4, 7, 5, 6],
        'TLBalHCPct': [0.8, 0.85, 0.75, 0.9, 0.78, 0.82, 0.79, 0.88, 0.81, 0.86],
        'TLSatPct': [0.5, 0.4, 0.6, 0.3, 0.55, 0.45, 0.65, 0.35, 0.58, 0.48],
        'TLDel3060Cnt24': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        'TLDel90Cnt24': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'TLDel60CntAll': [2, 3, 2, 3, 2, 3, 2, 3, 2, 3],
        'TLOpenPct': [0.2, 0.15, 0.25, 0.1, 0.22, 0.18, 0.28, 0.12, 0.21, 0.17],
        'TLBadDerogCnt': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        'TLOpen24Pct': [0.3, 0.25, 0.35, 0.2, 0.32, 0.28, 0.38, 0.22, 0.31, 0.27],
        DUMMY_TARGET_COLUMN_NAME: [1] * 10
    })
], ignore_index=True)


# Define preprocessing steps
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), 
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing_category')), 
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, NUMERICAL_FEATURES),
        *((('cat', categorical_transformer, CATEGORICAL_FEATURES),) if CATEGORICAL_FEATURES else ())
    ])

# Create a pipeline: preprocess then apply MLPClassifier (Neural Network)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True, n_iter_no_change=50)) 
])

# "Fit" the pipeline on the dummy data
try:
    # Ensure dummy_features exist in dummy_data_for_training before fitting
    # Exclude 'ID' from features used for training the model
    dummy_features_for_training = [col for col in NUMERICAL_FEATURES + CATEGORICAL_FEATURES if col != 'ID']
    
    missing_dummy_features = [col for col in dummy_features_for_training if col not in dummy_data_for_training.columns]
    if missing_dummy_features:
        st.sidebar.error(f"Missing dummy features for model training: {missing_dummy_features}. Please check dummy_data_for_training.")
        st.stop()

    model_pipeline.fit(dummy_data_for_training[dummy_features_for_training], dummy_data_for_training[DUMMY_TARGET_COLUMN_NAME])
    st.sidebar.success("Simulated ML model (Neural Network) and preprocessor initialized for new data.")
except Exception as e:
    st.sidebar.error(f"Error initializing simulated ML model: {e}")
    st.stop() 

# --- Credit Scoring Function using the ML Model ---
@st.cache_data 
def get_credit_score_ml(df_input: pd.DataFrame, approval_threshold: int, actual_target_column: str): 
    """
    Applies the simulated ML model to score a DataFrame of credit applications.
    Performs robust NaN handling and type coercion before passing to the ML pipeline.
    """
    df_processed = df_input.copy()

    # Define features to be used for prediction (excluding the actual target column and 'ID')
    features_for_prediction = [col for col in NUMERICAL_FEATURES + CATEGORICAL_FEATURES if col != actual_target_column and col != 'ID']

    # Ensure all expected feature columns exist. Add if missing with default values.
    for col in features_for_prediction:
        if col not in df_processed.columns:
            if col in NUMERICAL_FEATURES:
                df_processed[col] = 0.0 
            elif col in CATEGORICAL_FEATURES:
                df_processed[col] = 'missing_category' 
    
    # Coerce numerical columns to numeric type, converting any non-numeric values to NaN
    for col in NUMERICAL_FEATURES:
        if col in df_processed.columns: # Only process if column exists after previous checks
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce') 
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean() if not df_processed[col].empty else 0.0) # Fill NaNs after conversion

    # Explicitly fill NaNs in categorical columns and ensure they are string type
    for col in CATEGORICAL_FEATURES:
        if col in df_processed.columns: # Only process if column exists after previous checks
            df_processed[col] = df_processed[col].astype(str).replace('nan', 'missing_category')
            df_processed[col] = df_processed[col].fillna('missing_category') 

    # Select and reorder columns to match the training order
    # Ensure only the actual features are passed to the model
    df_processed_for_prediction = df_processed[features_for_prediction]

    st.write("DEBUG: DataFrame info before ML prediction:")
    buffer = io.StringIO()
    df_processed_for_prediction.info(buf=buffer)
    st.text(buffer.getvalue())
    st.write(f"DEBUG: Total NaNs in data before prediction: {df_processed_for_prediction.isnull().sum().sum()}")


    probabilities = model_pipeline.predict_proba(df_processed_for_prediction)
    
    # Assuming class 0 (index 0) is "Good" and class 1 (index 1) is "Bad" (Default)
    # So, score is probability of NOT defaulting (Class 0)
    scores = probabilities[:, 0] * 100 

    decisions = np.where(scores >= approval_threshold, "Approved", "Rejected") 

    return pd.DataFrame({'Score': scores, 'Decision': decisions})


# --- LLM Integration for Explanations (MODIFIED TO BE SYNCHRONOUS) ---
def get_llm_explanation(features_dict: dict, score: float, decision: str):
    """
    Calls a Gemini LLM to get an explanation for a loan decision.
    This function is now synchronous and uses asyncio.run() internally for the fetch call.
    """
    prompt = f"""
    You are an expert credit risk analyst. Based on the following loan application features, 
    explain concisely why this loan received a score of {score:.2f} and was {decision}.
    Focus on 2-3 key factors from the provided features that likely influenced the decision.
    
    Loan Features:
    {json.dumps(features_dict, indent=2)}
    
    Provide a brief, clear explanation (max 50 words).
    """
    
    chatHistory = []
    chatHistory.append({ "role": "user", "parts": [{ "text": prompt }] }) # Corrected from .push to .append
    
    apiKey = "" 
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={apiKey}"
    
    try:
        # Define an inner async function to perform the actual fetch
        async def _fetch_explanation_async():
            response = await st.experimental_memo(fetch)( # `fetch` is assumed to be async
                apiUrl,
                method='POST',
                headers={'Content-Type': 'application/json'},
                body=json.dumps({"contents": chatHistory})
            )
            return await response.json()

        # Run the async fetch operation in the current event loop, or create a new one
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError: # No running loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result = loop.run_until_complete(_fetch_explanation_async())
        
        if result and result.get('candidates') and len(result['candidates']) > 0 and \
           result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts') and \
           len(result['candidates'][0]['content']['parts']) > 0:
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return "LLM could not generate an explanation."
    except Exception as e:
        st.error(f"Error calling LLM for explanation: {e}")
        return "Failed to get explanation from LLM."


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
    """
)

st.title("Credit Risk Dashboard (Powered by Neural Network with Adjustable Threshold!)") 
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

    # Target Column Selector (only show if a file is selected)
    if selected_s3_file_in_form:
        try:
            temp_obj = s3_client.get_object(Bucket=s3_bucket_name, Key=selected_s3_file_in_form)
            temp_df = pd.read_csv(io.BytesIO(temp_obj['Body'].read()))
            st.session_state['uploaded_df_columns'] = temp_df.columns.tolist()
            
            default_target_idx = 0
            # Check for 'default' (lowercase)
            if TARGET_COLUMN in st.session_state['uploaded_df_columns']:
                default_target_idx = st.session_state['uploaded_df_columns'].index(TARGET_COLUMN)
            elif len(st.session_state['uploaded_df_columns']) > 0:
                st.warning(f"Default target column '{TARGET_COLUMN}' not found. Please select the correct target column.")
                default_target_idx = 0 
            
            st.session_state['selected_target_column'] = st.selectbox(
                "Select the Target Column (e.g., 'default', 'TARGET'):", 
                options=st.session_state['uploaded_df_columns'],
                index=default_target_idx,
                key="target_column_selector"
            )
        except Exception as e:
            st.error(f"Error reading selected file to get columns for target selector: {e}")
            st.session_state['uploaded_df_columns'] = [] 
            st.session_state['selected_target_column'] = '' 
    else:
        st.session_state['uploaded_df_columns'] = [] # Clear columns if no file selected
        st.session_state['selected_target_column'] = '' # Clear selected target

    # This is the button that submits the form and triggers analysis
    analyze_submitted = st.form_submit_button(f"Analyze Selected File")

    if analyze_submitted:
        if selected_s3_file_in_form and st.session_state['selected_target_column']:
            st.write(f"DEBUG: Analyze form submitted. Selected file: `{selected_s3_file_in_form}`. Target column: `{st.session_state['selected_target_column']}`. Starting analysis...")
            st.session_state['selected_file_for_analysis'] = selected_s3_file_in_form 
            st.session_state['analyze_triggered'] = True 
            st.rerun() 
        else:
            st.warning("Please select a file and a target column to analyze.")
            st.session_state['analyze_triggered'] = False
    else:
        st.write("DEBUG: Analysis form not yet submitted.")

# --- Perform Analysis if Triggered ---
if st.session_state['analyze_triggered'] and st.session_state['selected_file_for_analysis']:
    file_to_analyze = st.session_state['selected_file_for_analysis']
    actual_target_col_name = st.session_state['selected_target_column']

    if not actual_target_col_name:
        st.error("No target column selected. Please select one from the dropdown.")
        st.session_state['analyze_triggered'] = False
        st.session_state['selected_file_for_analysis'] = None
        st.stop()

    st.write(f"DEBUG: Analysis triggered via session state for file: `{file_to_analyze}` with target column `{actual_target_col_name}`.")
    try:
        with st.spinner(f"Downloading and analyzing '{file_to_analyze}' from S3..."):
            obj = s3_client.get_object(Bucket=s3_bucket_name, Key=file_to_analyze)
            df = pd.read_csv(io.BytesIO(obj['Body'].read())) 
            st.write(f"DEBUG: Successfully read {len(df)} rows from CSV file.")
            st.write("DEBUG: Columns in loaded CSV file:", df.columns.tolist()) 

            # Ensure 'ID' column is string type for plotting if it exists
            # If 'ID' column is not present, create a unique identifier from index for plotting
            if 'ID' not in df.columns:
                df['ID'] = df.index.astype(str) + '_idx'
                st.write("DEBUG: 'ID' column not found, created unique IDs from index.")
            else:
                df['ID'] = df['ID'].astype(str)
                st.write("DEBUG: 'ID' column processed to string type.")

            st.write("DEBUG: Columns in df after ID processing:", df.columns.tolist()) 

            # Ensure the selected target column is present and is integer type for classification
            if actual_target_col_name in df.columns:
                df[actual_target_col_name] = pd.to_numeric(df[actual_target_col_name], errors='coerce').fillna(0).astype(int)
                st.write(f"DEBUG: '{actual_target_col_name}' column processed to integer type.")
            else:
                st.error(f"The selected target column '{actual_target_col_name}' was not found in the uploaded data. Please check your CSV and select the correct column.")
                st.session_state['scored_data'] = pd.DataFrame()
                st.session_state['analyze_triggered'] = False
                st.session_state['selected_file_for_analysis'] = None
                st.stop() 

            # Pass the actual_target_col_name to the scoring function
            results_df = get_credit_score_ml(df.copy(), st.session_state['approval_threshold'], actual_target_col_name) 
            
            # More robust way to ensure 'ID' is the first column for display
            if 'ID' in df.columns:
                # Get all columns, put 'ID' first, then the rest
                cols = ['ID'] + [col for col in df.columns if col != 'ID']
                df = df[cols]
                st.write("DEBUG: 'ID' column moved to front of DataFrame.")

            df_scored = pd.concat([df, results_df], axis=1)
            st.write(f"DEBUG: Columns in df_scored before dashboard display:", df_scored.columns.tolist()) 
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
    st.write("DEBUG: Columns in df_display at dashboard start:", df_display.columns.tolist()) 

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
        # Sort by Score descending and get the top 10
        top_10_approved_loans = df_display[df_display['Decision'] == 'Approved'].sort_values(by='Score', ascending=False).head(10)
        
        st.write(f"DEBUG: Columns in top_10_approved_loans: {top_10_approved_loans.columns.tolist()}") 
        st.write("DEBUG: Top 10 Approved Loans Data (ID and Score):")
        st.dataframe(top_10_approved_loans[['ID', 'Score']]) # Show relevant columns for debugging

        if not top_10_approved_loans.empty:
            # Ensure 'ID' is treated as a categorical type for sorting
            top_10_approved_loans['ID'] = top_10_approved_loans['ID'].astype(str)
            ordered_top_ids = top_10_approved_loans['ID'].tolist()

            fig_top10 = px.bar(
                top_10_approved_loans,
                x='ID', # Use 'ID' for x-axis
                y='Score',
                color_discrete_sequence=['#4CAF50'], 
                title='Top 10 Approved Applicants by Score', 
                labels={'ID': 'Applicant ID', 'Score': 'Credit Score'}, 
                hover_data=NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ['Decision'],
                text='Score' # Show score on bars
            )
            # Ensure bars are in descending order of score
            fig_top10.update_xaxes(categoryorder='array', categoryarray=ordered_top_ids, type='category') 
            fig_top10.update_traces(texttemplate='%{text:.2s}', textposition='outside') # Format text on bars
            fig_top10.update_layout(xaxis_tickangle=-45) 
            st.plotly_chart(fig_top10, use_container_width=True)
        else:
            st.info("No approved applicants to display in the top 10 chart based on current threshold.")


    with col2:
        st.subheader("Worst 10 Rejected Applicants (Lowest Scores)") 
        # Sort by Score ascending and get the top 10 (which are the worst)
        worst_10_rejected_loans = df_display[df_display['Decision'] == 'Rejected'].sort_values(by='Score', ascending=True).head(10)
        
        st.write(f"DEBUG: Columns in worst_10_rejected_loans: {worst_10_rejected_loans.columns.tolist()}") 
        st.write("DEBUG: Worst 10 Rejected Loans Data (ID and Score):")
        st.dataframe(worst_10_rejected_loans[['ID', 'Score']]) # Show relevant columns for debugging

        if not worst_10_rejected_loans.empty:
            # Get ordered IDs for Plotly category_orders
            worst_10_rejected_loans['ID'] = worst_10_rejected_loans['ID'].astype(str) # Ensure 'ID' is string
            ordered_worst_ids = worst_10_rejected_loans['ID'].tolist()

            fig_worst10 = px.bar(
                worst_10_rejected_loans,
                x='ID', # Use 'ID' for x-axis
                y='Score',
                color_discrete_sequence=['#f44336'], 
                title='Worst 10 Rejected Applicants by Score', 
                labels={'ID': 'Applicant ID', 'Score': 'Credit Score'}, 
                hover_data=NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ['Decision'],
                text='Score' # Show score on bars
            )
            # Ensure bars are in ascending order of score
            fig_worst10.update_xaxes(categoryorder='array', categoryarray=ordered_worst_ids, type='category') 
            fig_worst10.update_traces(texttemplate='%{text:.2s}', textposition='outside') # Format text on bars
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

    # --- New: Summary Statistics Section ---
    st.markdown("---")
    st.header("Overall Loan Summary & Average Statistics")
    
    total_loans = len(df_display)
    approved_count = len(df_display[df_display['Decision'] == 'Approved'])
    rejected_count = len(df_display[df_display['Decision'] == 'Rejected'])

    st.markdown(f"**Total Loans Analyzed:** {total_loans}")
    st.markdown(f"**Approved Loans:** {approved_count}")
    st.markdown(f"**Rejected Loans:** {rejected_count}")

    st.subheader("Average Statistics for Approved vs. Rejected Loans")

    # Select only numerical features for mean calculation
    features_for_stats = [col for col in NUMERICAL_FEATURES if col in df_display.columns]

    if not features_for_stats:
        st.info("No numerical features available for calculating average statistics.")
    else:
        approved_stats = df_display[df_display['Decision'] == 'Approved'][features_for_stats].mean().rename('Approved Avg')
        rejected_stats = df_display[df_display['Decision'] == 'Rejected'][features_for_stats].mean().rename('Rejected Avg')

        # Combine into a single DataFrame for display
        summary_stats_df = pd.DataFrame([approved_stats, rejected_stats]).T
        summary_stats_df.index.name = 'Feature'
        st.dataframe(summary_stats_df.style.format("{:.2f}"))

        st.info("Compare the average values of key features between approved and rejected loans to understand influencing factors.")

    # --- New: LLM Suggestions for Rejected Loans ---
    st.markdown("---")
    st.header("AI-Powered Explanations for Rejected Loans")
    st.info("Click on a rejected loan's ID to get an AI-generated explanation for the decision.")

    if not worst_10_rejected_loans.empty:
        for index, row in worst_10_rejected_loans.iterrows():
            loan_id = row['ID']
            score = row['Score']
            decision = row['Decision']
            
            # Extract relevant features for the LLM prompt
            # Convert Series to dictionary for easier LLM input
            loan_features = row[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].dropna().to_dict()

            with st.expander(f"Explain why Loan ID: **{loan_id}** (Score: {score:.2f}, Decision: {decision}) was rejected"):
                # Use st.spinner for synchronous LLM call
                with st.spinner(f"Generating explanation for {loan_id}..."):
                    explanation = get_llm_explanation(loan_features, score, decision) # No 'await' here
                    st.markdown(explanation)
    else:
        st.info("No rejected loans to generate explanations for at the current threshold.")


else:
    st.write("DEBUG: No scored data found in session state or data is empty. Dashboard not displayed.")

# --- AWS and Langchain Integration Explanation (Recap) ---
st.markdown("---")
st.subheader("How this app would integrate with AWS and Langchain (Recap):")

st.markdown(
    """
    This Streamlit app now demonstrates uploading and analyzing **sophisticated credit data** from AWS S3, leveraging a powerful Neural Network machine learning model.

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
