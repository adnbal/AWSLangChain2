import streamlit as st
import pandas as pd
import numpy as np
import boto3
import io
import datetime
import os
import time
import json
import requests # For making HTTP requests to the LLM API

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

# Define numerical features specifically for plotting average statistics (excluding TLTimeFirst)
NUMERICAL_FEATURES_FOR_PLOTTING = [
    'DerogCnt', 'CollectCnt', 'BanruptcyInd', 'InqCnt06', 'InqTimeLast', 'InqFinanceCnt24',
    # 'TLTimeFirst' is excluded from plotting here to improve scaling
    'TLTimeLast', 'TLCnt03', 'TLCnt12', 'TLCnt24', 'TLCnt',
    'TLSum', 'TLMaxSum', 'TLSatCnt', 'TLDel60Cnt', 'TLBadCnt24',
    'TL75UtilCnt', 'TL50UtilCnt', 'TLBalHCPct', 'TLSatPct', 'TLDel3060Cnt24',
    'TLDel90Cnt24', 'TLDel60CntAll', 'TLOpenPct', 'TLBadDerogCnt', 'TLOpen24Pct'
]

# Group numerical features for better visualization scaling
GROUP_1_COUNTS_INDICATORS = [
    'DerogCnt', 'CollectCnt', 'BanruptcyInd', 'InqCnt06', 'InqFinanceCnt24',
    'TLCnt03', 'TLCnt12', 'TLCnt24', 'TLCnt', 'TLSatCnt', 'TLDel60Cnt',
    'TLBadCnt24', 'TL75UtilCnt', 'TL50UtilCnt', 'TLDel3060Cnt24',
    'TLDel90Cnt24', 'TLDel60CntAll', 'TLBadDerogCnt'
]

GROUP_2_TIME_PERCENTAGES = [
    'InqTimeLast', 'TLTimeLast', 'TLBalHCPct', 'TLSatPct', 'TLOpenPct', 'TLOpen24Pct'
]

GROUP_3_SUMS_MAXSUMS = [
    'TLSum', 'TLMaxSum'
]


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


# --- LLM Integration for Explanations (MODIFIED TO USE OpenAI API) ---
def get_llm_explanation(features_dict: dict, score: float, decision: str):
    """
    Calls an OpenAI LLM to get an explanation for a loan decision using requests.
    """
    prompt = f"""
    You are an expert credit risk analyst. Based on the following loan application features, 
    explain concisely why this loan received a score of {score:.2f} and was {decision}.
    Focus on 2-3 key factors from the provided features that likely influenced the decision.
    
    Loan Features:
    {json.dumps(features_dict, indent=2)}
    
    Provide a brief, clear explanation (max 50 words).
    """
    
    try:
        # Get OpenAI API key from Streamlit secrets, accessing the nested structure
        openai_api_key = st.secrets["openai"]["api_key"]
        
        # OpenAI API endpoint for chat completions
        apiUrl = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
        
        payload = {
            "model": "gpt-3.5-turbo", # You can change this to 'gpt-4' or other models if available
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100, # Limit response length
            "temperature": 0.7 # Control creativity
        }
        
        response = requests.post(
            apiUrl,
            headers=headers,
            json=payload
        )
        response.raise_for_status() # Raise an exception for HTTP errors (e.g., 4xx or 5xx)
        
        result = response.json()
        
        # OpenAI response parsing
        if result and result.get('choices') and len(result['choices']) > 0 and \
           result['choices'][0].get('message') and result['choices'][0]['message'].get('content'):
            return result['choices'][0]['message']['content']
        else:
            return "LLM could not generate an explanation."
    except KeyError:
        st.error("OpenAI API Key not found in Streamlit secrets. Please ensure 'openai.api_key' is set in your secrets.toml or Streamlit Cloud secrets.")
        return "Failed to get explanation from LLM (API Key missing or incorrectly configured)."
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling LLM for explanation: {e}")
        return "Failed to get explanation from LLM."
    except json.JSONDecodeError:
        st.error("Failed to decode JSON response from LLM API.")
        return "Failed to get explanation from LLM (JSON decode error)."
    except Exception as e:
        st.error(f"An unexpected error occurred while calling LLM: {e}")
        return "Failed to get explanation from LLM."


# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Credit Scoring Dashboard with S3", layout="wide")

# Custom CSS for styling the Streamlit app to match the dark theme and card layout
st.markdown(
    """
    <style>
    /* Overall Dark Theme */
    body {
        background-color: #1a1a2e; /* Dark blue background */
        color: #e0e0e0; /* Light grey text */
        font-family: 'Inter', sans-serif; /* Use Inter font */
    }
    .main {
        background-color: #1a1a2e; /* Dark blue background for main content */
        padding: 0px; /* Remove padding from main to allow full width cards */
    }
    .stApp {
        background-color: #1a1a2e; /* Ensure app background is dark */
    }

    /* Header and Subheader Styling */
    h1, h2, h3, h4, h5, h6 {
        color: #e0e0e0; /* Light grey for all headers */
    }
    .stMarkdown h1 {
        text-align: center;
        color: #e0e0e0;
        padding-top: 20px;
        padding-bottom: 10px;
    }
    .stMarkdown h2, .stMarkdown h3 {
        padding-top: 15px;
        padding-bottom: 5px;
    }

    /* Card Styling */
    .dashboard-card {
        background-color: #2a2a4a; /* Slightly lighter dark blue for cards */
        border-radius: 12px; /* Rounded corners */
        padding: 20px;
        margin: 10px; /* Space between cards */
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3); /* Soft shadow */
        border: 1px solid #3f3f6f; /* Subtle border */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 180px; /* Ensure cards have a minimum height */
    }
    .metric-card {
        min-height: 150px; /* Smaller height for top row metric cards */
    }
    .chart-card {
        min-height: 300px; /* Taller height for chart cards */
    }
    .table-card {
        min-height: 350px; /* Taller height for table cards */
    }

    /* Metric Value Display */
    .metric-value {
        font-size: 2.2em;
        font-weight: bold;
        color: #6a8dff; /* Blue for primary metrics */
        margin-bottom: 5px;
    }
    .metric-change {
        font-size: 0.9em;
        color: #a0a0a0; /* Grey for secondary info */
    }
    .positive-change {
        color: #4CAF50; /* Green for positive change */
    }
    .negative-change {
        color: #f44336; /* Red for negative change */
    }
    .metric-title {
        font-size: 1.1em;
        font-weight: bold;
        color: #e0e0e0;
        margin-bottom: 10px;
    }
    .info-icon {
        float: right;
        color: #a0a0a0;
        font-size: 0.9em;
        cursor: pointer;
    }

    /* Streamlit Widget Styling */
    .stButton>button {
        background-color: #6a8dff; /* Blue for buttons */
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #537be6;
        transform: translateY(-2px);
    }
    .stTextInput>div>div>input, .stSelectbox>div>div {
        background-color: #3f3f6f; /* Darker background for inputs */
        color: #e0e0e0;
        border-radius: 8px;
        border: 1px solid #5a5a8a;
        padding: 8px;
    }
    .stSlider>div>div>div>div {
        background-color: #6a8dff; /* Blue for slider track */
    }
    .stSlider [data-baseweb="slider"] {
        background-color: #3f3f6f; /* Darker background for slider rail */
    }
    .stRadio > label {
        color: #e0e0e0; /* Radio button labels */
    }

    /* Info/Success/Warning Boxes */
    .stAlert {
        border-radius: 8px;
    }
    .stAlert.info {
        background-color: #2a2a4a;
        color: #6a8dff;
        border-left: 5px solid #6a8dff;
    }
    .stAlert.success {
        background-color: #2a2a4a;
        color: #4CAF50;
        border-left: 5px solid #4CAF50;
    }
    .stAlert.warning {
        background-color: #2a2a4a;
        color: #FFC107;
        border-left: 5px solid #FFC107;
    }
    .stAlert.error {
        background-color: #2a2a4a;
        color: #f44336;
        border-left: 5px solid #f44336;
    }

    /* Plotly Chart Background */
    .js-plotly-plot {
        background-color: #2a2a4a !important; /* Match card background */
        border-radius: 12px; /* Apply border-radius to plot area */
    }
    .modebar {
        background-color: #2a2a4a !important; /* Match card background */
        border-radius: 0 0 12px 12px; /* Rounded bottom corners */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Helper function to create a styled card ---
def create_card(title, content, card_type="dashboard-card"):
    st.markdown(f'<div class="{card_type}"><h3>{title}</h3>{content}</div>', unsafe_allow_html=True)

# --- Helper function for metric cards ---
def create_metric_card(title, current_value, prev_value, unit="", is_percentage=False, chart_data=None, chart_y_col=None):
    if prev_value is not None and prev_value != 0:
        change = ((current_value - prev_value) / prev_value) * 100
        change_text = f"{change:.2f}%"
        change_class = "positive-change" if change >= 0 else "negative-change"
        arrow = "▲" if change >= 0 else "▼"
    else:
        change_text = "N/A"
        change_class = ""
        arrow = ""

    current_display = f"{current_value:.2f}{unit}" if not is_percentage else f"{current_value:.2f}%"
    prev_display = f"{prev_value:.2f}{unit}" if not is_percentage else f"{prev_value:.2f}%"

    card_content = f"""
    <div class="metric-title">{title} <span class="info-icon">ⓘ</span></div>
    <div class="metric-value">{current_display}</div>
    <div class="metric-change">
        <span class="{change_class}">{arrow} {change_text}</span> vs {prev_display}
    </div>
    """
    
    # Add a placeholder for a sparkline chart if data is provided
    if chart_data is not None and chart_y_col is not None:
        # Create a tiny Plotly chart for the sparkline effect
        fig = px.line(
            chart_data, 
            y=chart_y_col, 
            height=50, 
            width=200, # Adjust width to fit card
            template="plotly_dark"
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            xaxis_visible=False,
            yaxis_visible=False,
            hovermode=False,
            transition_duration=300
        )
        fig.update_traces(mode='lines', line=dict(width=2, color='#6a8dff')) # Blue line
        
        # Convert to HTML and embed
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        card_content += f'<div style="width: 100%; overflow: hidden;">{chart_html}</div>'

    st.markdown(f'<div class="dashboard-card metric-card">{card_content}</div>', unsafe_allow_html=True)


# --- Dashboard Header ---
st.markdown(
    """
    <div style="background-color: #0f0f20; padding: 15px 20px; border-radius: 12px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
        <h2 style="color: #e0e0e0; margin: 0;">Credit Risk Analysis Dashboard</h2>
        <div>
            <button style="background-color: #6a8dff; color: white; padding: 8px 15px; border-radius: 8px; border: none; margin-left: 10px;">Synopsis</button>
            <button style="background-color: #6a8dff; color: white; padding: 8px 15px; border-radius: 8px; border: none; margin-left: 10px;">Regional Analysis</button>
            <button style="background-color: #6a8dff; color: white; padding: 8px 15px; border-radius: 8px; border: none; margin-left: 10px;">Orders</button>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- File Upload and S3 Integration ---
st.sidebar.header("Data Management")
s3_client = None
s3_bucket_name = None
aws_region_name = None

try:
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

    st.sidebar.markdown("---")
    st.sidebar.subheader("AWS Secrets Status:")
    st.sidebar.write(f"AWS Access Key ID: `{'*' * (len(aws_access_key) - 4)}{aws_access_key[-4:]}`")
    st.sidebar.write(f"AWS Secret Access Key: `{'*' * (len(aws_secret_key) - 4)}{aws_secret_key[-4:]}`")
    st.sidebar.write(f"S3 Bucket Name: `{s3_bucket_name}`")
    st.sidebar.write(f"AWS Region: `{aws_region_name}`")

except KeyError as e:
    st.sidebar.error(f"Secret key not found: {e}. Please ensure your secrets are configured correctly as nested keys under `[aws]` in Streamlit Cloud or `.streamlit/secrets.toml`.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"An error occurred while initializing S3 client: {e}")
    st.stop()

def clear_file_uploader():
    st.session_state['file_uploader_key'] += 1

st.sidebar.subheader("Upload Credit Data to S3") 
uploaded_file = st.sidebar.file_uploader(
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
        st.sidebar.success(f"File '{file_name}' uploaded successfully to S3 as '{s3_file_key}'!")
        st.session_state['last_uploaded_s3_key'] = s3_file_key
        st.session_state['analyze_triggered'] = False 
        st.session_state['scored_data'] = pd.DataFrame() 
        st.sidebar.info("File uploaded. Please select it from the dropdown below and click 'Analyze'.")
        time.sleep(1) 

        clear_file_uploader() 
        st.rerun() 

    except Exception as e:
        st.sidebar.error(f"Error uploading file to S3: {e}")

st.sidebar.subheader("Analyze Credit Data from S3") 

s3_files = []
try:
    if s3_client:
        response = s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix="uploads/")
        if 'Contents' in response:
            s3_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.csv')] 
            s3_files.sort(reverse=True)
except Exception as e:
    st.sidebar.warning(f"Could not list files from S3 bucket: {e}. Ensure 'uploads/' prefix exists or bucket is not empty, and permissions are correct.")
    s3_files = []

with st.sidebar.form("analysis_trigger_form"):
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

    if selected_s3_file_in_form:
        try:
            temp_obj = s3_client.get_object(Bucket=s3_bucket_name, Key=selected_s3_file_in_form)
            temp_df = pd.read_csv(io.BytesIO(temp_obj['Body'].read()))
            st.session_state['uploaded_df_columns'] = temp_df.columns.tolist()
            
            default_target_idx = 0
            if TARGET_COLUMN in st.session_state['uploaded_df_columns']:
                default_target_idx = st.session_state['uploaded_df_columns'].index(TARGET_COLUMN)
            elif len(st.session_state['uploaded_df_columns']) > 0:
                st.sidebar.warning(f"Default target column '{TARGET_COLUMN}' not found. Please select the correct target column.")
                default_target_idx = 0 
            
            st.session_state['selected_target_column'] = st.selectbox(
                "Select the Target Column (e.g., 'default', 'TARGET'):", 
                options=st.session_state['uploaded_df_columns'],
                index=default_target_idx,
                key="target_column_selector"
            )
        except Exception as e:
            st.sidebar.error(f"Error reading selected file to get columns for target selector: {e}")
            st.session_state['uploaded_df_columns'] = [] 
            st.session_state['selected_target_column'] = '' 
    else:
        st.session_state['uploaded_df_columns'] = [] 
        st.session_state['selected_target_column'] = '' 

    analyze_submitted = st.form_submit_button(f"Analyze Selected File")

    if analyze_submitted:
        if selected_s3_file_in_form and st.session_state['selected_target_column']:
            st.session_state['selected_file_for_analysis'] = selected_s3_file_in_form 
            st.session_state['analyze_triggered'] = True 
            st.rerun() 
        else:
            st.sidebar.warning("Please select a file and a target column to analyze.")
            st.session_state['analyze_triggered'] = False

if st.session_state['scored_data'] is not None and not st.session_state['scored_data'].empty:
    if st.sidebar.button("Clear Displayed Data", key="clear_data_button_sidebar"):
        st.session_state['scored_data'] = pd.DataFrame()
        st.session_state['analyze_triggered'] = False
        st.session_state['selected_file_for_analysis'] = None
        st.sidebar.success("Displayed data cleared.")
        st.rerun() 

# --- Main Dashboard Content ---
if st.session_state['analyze_triggered'] and st.session_state['selected_file_for_analysis']:
    file_to_analyze = st.session_state['selected_file_for_analysis']
    actual_target_col_name = st.session_state['selected_target_column']

    if not actual_target_col_name:
        st.error("No target column selected. Please select one from the dropdown.")
        st.session_state['analyze_triggered'] = False
        st.session_state['selected_file_for_analysis'] = None
        st.stop()

    try:
        with st.spinner(f"Downloading and analyzing '{file_to_analyze}' from S3..."):
            obj = s3_client.get_object(Bucket=s3_bucket_name, Key=file_to_analyze)
            df = pd.read_csv(io.BytesIO(obj['Body'].read())) 

            if 'ID' not in df.columns:
                df['ID'] = df.index.astype(str) + '_idx'
            else:
                df['ID'] = df['ID'].astype(str)

            if actual_target_col_name in df.columns:
                df[actual_target_col_name] = pd.to_numeric(df[actual_target_col_name], errors='coerce').fillna(0).astype(int)
            else:
                st.error(f"The selected target column '{actual_target_col_name}' was not found in the uploaded data. Please check your CSV and select the correct column.")
                st.session_state['scored_data'] = pd.DataFrame()
                st.session_state['analyze_triggered'] = False
                st.session_state['selected_file_for_analysis'] = None
                st.stop() 

            results_df = get_credit_score_ml(df.copy(), st.session_state['approval_threshold'], actual_target_col_name) 
            
            if 'ID' in df.columns:
                cols = ['ID'] + [col for col in df.columns if col != 'ID']
                df = df[cols]

            df_scored = pd.concat([df, results_df], axis=1)
            st.session_state['scored_data'] = df_scored
            st.success(f"Analysis complete for '{file_to_analyze}'.")
            st.session_state['analyze_triggered'] = False 
            st.session_state['selected_file_for_analysis'] = None 

    except Exception as e:
        st.error(f"Error analyzing file from S3: {e}. Please check file format and column names in your CSV file.")
        st.session_state['scored_data'] = pd.DataFrame()
        st.session_state['analyze_triggered'] = False 
        st.session_state['selected_file_for_analysis'] = None 

if 'scored_data' in st.session_state and not st.session_state['scored_data'].empty:
    df_display = st.session_state['scored_data']

    # --- Top Row Metrics ---
    st.markdown("### Key Credit Metrics")
    col_metrics = st.columns(4)

    # Dummy data for sparklines (replace with actual time series if available)
    # For demonstration, let's create a simple dummy time series based on the number of rows
    num_rows = len(df_display)
    dummy_dates = pd.date_range(start='2024-01-01', periods=num_rows, freq='D') # Daily data
    dummy_trend_data = pd.DataFrame({
        'Date': dummy_dates,
        'Value': np.linspace(50, 100, num_rows) + np.random.randn(num_rows) * 5 # Simple increasing trend with noise
    })
    # Take a sample for sparkline if data is too large
    sparkline_data_sample = dummy_trend_data.sample(min(50, len(dummy_trend_data))).sort_values('Date')

    # Ensure numerical columns are truly numeric before aggregation
    df_display_numeric = df_display.copy()
    for col in ['TLSum', 'Score', 'TLCnt', 'TLBadCnt24']:
        if col in df_display_numeric.columns:
            df_display_numeric[col] = pd.to_numeric(df_display_numeric[col], errors='coerce').fillna(0)

    with col_metrics[0]:
        current_tlsum = df_display_numeric['TLSum'].sum() if 'TLSum' in df_display_numeric.columns else 0
        prev_tlsum = current_tlsum * 0.95 # Dummy previous year
        create_metric_card("Total Loan Sum", current_tlsum, prev_tlsum, unit="", chart_data=sparkline_data_sample, chart_y_col='Value')

    with col_metrics[1]:
        current_score_avg = df_display_numeric['Score'].mean() if 'Score' in df_display_numeric.columns else 0
        prev_score_avg = current_score_avg * 1.02 # Dummy previous year
        create_metric_card("Avg Credit Score", current_score_avg, prev_score_avg, is_percentage=False, chart_data=sparkline_data_sample, chart_y_col='Value')

    with col_metrics[2]:
        current_tlcnt = df_display_numeric['TLCnt'].sum() if 'TLCnt' in df_display_numeric.columns else 0
        prev_tlcnt = current_tlcnt * 0.9 # Dummy previous year
        create_metric_card("Total Trades", current_tlcnt, prev_tlcnt, unit="", chart_data=sparkline_data_sample, chart_y_col='Value')

    with col_metrics[3]:
        current_bad_cnt = df_display_numeric['TLBadCnt24'].sum() if 'TLBadCnt24' in df_display_numeric.columns else 0
        prev_bad_cnt = current_bad_cnt * 1.1 # Dummy previous year (increased bad count)
        create_metric_card("Bad Trades (24M)", current_bad_cnt, prev_bad_cnt, unit="", chart_data=sparkline_data_sample, chart_y_col='Value')


    # --- Middle Row Charts ---
    st.markdown("### Performance Overview")
    col_middle_charts = st.columns(2)

    with col_middle_charts[0]:
        st.markdown('<div class="dashboard-card chart-card">', unsafe_allow_html=True)
        st.markdown("<h4>Credit Decision Trend (Dummy Years)</h4>")
        # Dummy data for trend over years
        trend_data = pd.DataFrame({
            'Year': [2021, 2022, 2023, 2024],
            'Approved': [np.random.randint(100, 500), np.random.randint(100, 500), np.random.randint(100, 500), np.random.randint(100, 500)],
            'Rejected': [np.random.randint(10, 100), np.random.randint(10, 100), np.random.randint(10, 100), np.random.randint(10, 100)]
        })
        fig_trend = px.bar(
            trend_data.melt(id_vars='Year', var_name='Decision', value_name='Count'),
            x='Year', y='Count', color='Decision',
            barmode='group',
            color_discrete_map={'Approved': '#6a8dff', 'Rejected': '#f44336'},
            template="plotly_dark",
            text='Count'
        )
        fig_trend.update_layout(xaxis_tickangle=-45, transition_duration=500,
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font_color="#e0e0e0", margin=dict(t=30, b=0, l=0, r=0))
        fig_trend.update_traces(texttemplate='%{y}', textposition='outside')
        st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_middle_charts[1]:
        st.markdown('<div class="dashboard-card table-card">', unsafe_allow_html=True)
        st.markdown("<h4>Business Overview by Decision</h4>")
        # Aggregate data by Decision for overview
        overview_df = df_display.groupby('Decision').agg(
            Total_Loans=('ID', 'count'),
            Avg_Score=('Score', 'mean'),
            Total_TLSum=('TLSum', 'sum'),
            Total_BadCnt24=('TLBadCnt24', 'sum')
        ).reset_index()
        overview_df.columns = ['Decision', 'Total Loans', 'Avg Score', 'Total Loan Sum', 'Total Bad Trades']
        st.dataframe(overview_df.style.format({
            'Avg Score': "{:.2f}",
            'Total Loan Sum': "{:,.0f}",
            'Total Bad Trades': "{:,.0f}"
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#3f3f6f'), ('color', '#e0e0e0')]},
            {'selector': 'td', 'props': [('background-color', '#2a2a4a'), ('color', '#e0e0e0')]},
            {'selector': 'tr:nth-of-type(odd)', 'props': [('background-color', '#2a2a4a')]},
            {'selector': 'tr:nth-of-type(even)', 'props': [('background-color', '#20203a')]}
        ]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Bottom Row Charts ---
    st.markdown("### Detailed Insights")
    col_bottom_charts = st.columns(2)

    with col_bottom_charts[0]:
        st.markdown('<div class="dashboard-card chart-card">', unsafe_allow_html=True)
        st.markdown("<h4>Decision Contribution by Derogatory Count Group</h4>")
        # Create a categorical feature from DerogCnt for pie chart
        df_display['DerogCnt_Group'] = pd.cut(df_display['DerogCnt'], bins=[-1, 0, 1, 5, df_display['DerogCnt'].max()],
                                              labels=['No Derog', '1 Derog', '2-5 Derogs', '>5 Derogs'],
                                              right=True, include_lowest=True)
        
        pie_data = df_display.groupby('DerogCnt_Group')['ID'].count().reset_index()
        pie_data.columns = ['DerogCnt_Group', 'Count']
        
        fig_pie = px.pie(
            pie_data,
            values='Count',
            names='DerogCnt_Group',
            title='Proportion of Loans by Derogatory Count Group',
            color_discrete_sequence=px.colors.sequential.RdBu, # A diverging color scale
            template="plotly_dark"
        )
        fig_pie.update_layout(transition_duration=500,
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color="#e0e0e0", margin=dict(t=30, b=0, l=0, r=0))
        fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#2a2a4a', width=2)))
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_bottom_charts[1]:
        st.markdown('<div class="dashboard-card chart-card">', unsafe_allow_html=True)
        st.markdown("<h4>Average Statistics by Loan Decision (Group 1: Counts & Indicators)</h4>")
        features_group1 = [col for col in GROUP_1_COUNTS_INDICATORS if col in df_display.columns]
        if features_group1:
            df_numeric_for_stats = df_display[features_group1 + ['Decision']].apply(pd.to_numeric, errors='coerce').fillna(0)
            summary_stats_group1 = df_numeric_for_stats.groupby('Decision').mean().T.reset_index()
            summary_stats_group1.columns = ['Feature', 'Approved Avg', 'Rejected Avg']
            summary_stats_long_group1 = summary_stats_group1.melt(
                id_vars='Feature', var_name='Decision Type', value_name='Average Value'
            )
            fig_group1 = px.bar(
                summary_stats_long_group1,
                x='Feature', y='Average Value', color='Decision Type', barmode='group',
                labels={'Feature': 'Credit Feature', 'Average Value': 'Average Value'},
                color_discrete_map={'Approved Avg': '#6a8dff', 'Rejected Avg': '#f44336'},
                text_auto='.2s',
                template="plotly_dark"
            )
            fig_group1.update_layout(xaxis_tickangle=-45, transition_duration=500,
                                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                     font_color="#e0e0e0", margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_group1, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No features in Group 1 available for plotting.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Additional charts for other groups (similar structure)
    st.markdown("### Further Average Statistics")
    col_avg_charts = st.columns(2)

    with col_avg_charts[0]:
        st.markdown('<div class="dashboard-card chart-card">', unsafe_allow_html=True)
        st.markdown("<h4>Average Statistics by Loan Decision (Group 2: Time & Percentages)</h4>")
        features_group2 = [col for col in GROUP_2_TIME_PERCENTAGES if col in df_display.columns]
        if features_group2:
            df_numeric_for_stats = df_display[features_group2 + ['Decision']].apply(pd.to_numeric, errors='coerce').fillna(0)
            summary_stats_group2 = df_numeric_for_stats.groupby('Decision').mean().T.reset_index()
            summary_stats_group2.columns = ['Feature', 'Approved Avg', 'Rejected Avg']
            summary_stats_long_group2 = summary_stats_group2.melt(
                id_vars='Feature', var_name='Decision Type', value_name='Average Value'
            )
            fig_group2 = px.bar(
                summary_stats_long_group2,
                x='Feature', y='Average Value', color='Decision Type', barmode='group',
                labels={'Feature': 'Credit Feature', 'Average Value': 'Average Value'},
                color_discrete_map={'Approved Avg': '#6a8dff', 'Rejected Avg': '#f44336'},
                text_auto='.2s',
                template="plotly_dark"
            )
            fig_group2.update_layout(xaxis_tickangle=-45, transition_duration=500,
                                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                     font_color="#e0e0e0", margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_group2, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No features in Group 2 available for plotting.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_avg_charts[1]:
        st.markdown('<div class="dashboard-card chart-card">', unsafe_allow_html=True)
        st.markdown("<h4>Average Statistics by Loan Decision (Group 3: Sums & Max Sums)</h4>")
        features_group3 = [col for col in GROUP_3_SUMS_MAXSUMS if col in df_display.columns]
        if features_group3:
            df_numeric_for_stats = df_display[features_group3 + ['Decision']].apply(pd.to_numeric, errors='coerce').fillna(0)
            summary_stats_group3 = df_numeric_for_stats.groupby('Decision').mean().T.reset_index()
            summary_stats_group3.columns = ['Feature', 'Approved Avg', 'Rejected Avg']
            summary_stats_long_group3 = summary_stats_group3.melt(
                id_vars='Feature', var_name='Decision Type', value_name='Average Value'
            )
            fig_group3 = px.bar(
                summary_stats_long_group3,
                x='Feature', y='Average Value', color='Decision Type', barmode='group',
                labels={'Feature': 'Credit Feature', 'Average Value': 'Average Value'},
                color_discrete_map={'Approved Avg': '#6a8dff', 'Rejected Avg': '#f44336'},
                text_auto='.2s',
                template="plotly_dark"
            )
            fig_group3.update_layout(xaxis_tickangle=-45, transition_duration=500,
                                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                     font_color="#e0e0e0", margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_group3, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No features in Group 3 available for plotting.")
        st.markdown('</div>', unsafe_allow_html=True)


    # --- Full Data Table (optional, can be moved to a separate tab/section) ---
    st.markdown("### Full Credit Data")
    st.markdown('<div class="dashboard-card table-card">', unsafe_allow_html=True)
    st.dataframe(df_display.style.set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#3f3f6f'), ('color', '#e0e0e0')]},
            {'selector': 'td', 'props': [('background-color', '#2a2a4a'), ('color', '#e0e0e0')]},
            {'selector': 'tr:nth-of-type(odd)', 'props': [('background-color', '#2a2a4a')]},
            {'selector': 'tr:nth-of-type(even)', 'props': [('background-color', '#20203a')]}
        ]), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- LLM Explanations ---
    st.markdown("---")
    st.header("AI-Powered Explanations for Rejected Loans")
    st.info("Click on a rejected loan's ID to get an AI-generated explanation for the decision.")

    vulnerable_loans = df_display[df_display['Decision'] == 'Rejected'].sort_values(by='Score', ascending=True)

    if not vulnerable_loans.empty:
        for index, row in vulnerable_loans.iterrows():
            loan_id = row['ID']
            score = row['Score']
            decision = row['Decision']
            
            loan_features = row[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].dropna().to_dict()

            with st.expander(f"Explain why Loan ID: **{loan_id}** (Score: {score:.2f}, Decision: {decision}) was rejected"):
                with st.spinner(f"Generating explanation for {loan_id}..."):
                    explanation = get_llm_explanation(loan_features, score, decision) 
                    st.markdown(explanation)
    else:
        st.info("No rejected loans to generate explanations for at the current threshold.")


else:
    st.info("Upload a CSV file and click 'Analyze' to view the dashboard.")

# --- AWS and LLM Integration Explanation (Recap) ---
st.markdown("---")
st.subheader("How this app would integrate with AWS and LLM (Recap):")

st.markdown(
    """
    This Streamlit app now demonstrates uploading and analyzing **sophisticated credit data** from AWS S3, leveraging a powerful Neural Network machine learning model.

    ### ☁️ AWS Cloud Integration:
    * **Data Storage (S3):** Data files are stored in AWS S3, providing durable and scalable storage.
    * **Machine Learning Model Hosting (SageMaker):** In a real scenario, the `get_credit_score_ml` function would call a sophisticated ML model deployed on AWS SageMaker for more accurate and robust predictions. This separates the heavy computation from the Streamlit app.
    * **Serverless Functions (Lambda):** Could be used for automated processing of new files uploaded to S3 (e.g., triggering the scoring process automatically).
    * **Authentication & Authorization (Cognito):** For secure user access to the app and S3, ensuring only authorized users can upload or view sensitive data.
    * **Logging & Monitoring (CloudWatch):** To track app performance, S3 interactions, and potential errors, providing insights for operational management.

    ### 🧠 Large Language Model (LLM) Integration:
    The LLM (currently OpenAI's GPT-3.5-turbo) is used to provide **Explainable AI (XAI)**. After identifying rejected applicants, the LLM generates concise, human-readable explanations for *why* specific applicants were flagged as high-risk, based on their input features. This helps in understanding and communicating complex model decisions.
    """
)
