import streamlit as st
import pandas as pd
import numpy as np
import boto3
import io
import datetime
import os
import time

# Import scikit-learn components
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 

# Import Plotly for visualizations
import plotly.express as px

# --- Initialize session state variables using setdefault for robustness ---
st.session_state.setdefault('scored_data', pd.DataFrame())
st.session_state.setdefault('last_uploaded_s3_key', None)
st.session_state.setdefault('file_uploader_key', 0) 
st.session_state.setdefault('approval_threshold', 50) # Default approval threshold
st.session_state.setdefault('uploaded_df_columns', []) # To store columns of the last uploaded file
st.session_state.setdefault('selected_target_column', 'TARGET') # Default target column name is 'TARGET' for credit.csv

# --- Define Feature Columns for the 'credit.csv' dataset ---
# This list MUST match the features the model expects, and accounts for the duplicate column in your CSV.
NUMERICAL_FEATURES = [
    'DerogCnt', 'CollectCnt', 'BanruptcyInd', 'InqCnt06', 'InqTimeLast', 'InqFinanceCnt24',
    'TLTimeFirst', 'TLTimeLast', 'TLCnt03', 'TLCnt12', 'TLCnt24', 'TLCnt',
    'TLSum', 'TLMaxSum', 'TLSatCnt', 'TLDel60Cnt', 'TLDel60Cnt24', # Corrected: TLDel60Cnt24 is here once
    'TLBadCnt24', 'TL75UtilCnt', 'TL50UtilCnt', 'TLBalHCPct', 'TLSatPct', 'TLDel3060Cnt24',
    'TLDel90Cnt24', 'TLDel60CntAll', 'TLOpenPct', 'TLBadDerogCnt', 'TLOpen24Pct'
]
CATEGORICAL_FEATURES = [] # No explicit categorical features based on previous analysis

# Define numerical features specifically for plotting average statistics (excluding TLTimeFirst)
# These groups were introduced to address scaling issues in the "Average Statistics" graph
GROUP_1_COUNTS_INDICATORS = [
    'DerogCnt', 'CollectCnt', 'BanruptcyInd', 'InqCnt06', 'InqFinanceCnt24',
    'TLCnt03', 'TLCnt12', 'TLCnt24', 'TLCnt', 'TLSatCnt', 'TLDel60Cnt',
    'TLBadCnt24', 'TL75UtilCnt', 'TL50UtilCnt', 'TLDel3060Cnt24',
    'TLDel90Cnt24', 'TLDel60CntAll', 'TLBadDerogCnt', 'TLDel60Cnt24' # Added TLDel60Cnt24 here for plotting
]

GROUP_2_TIME_PERCENTAGES = [
    'InqTimeLast', 'TLTimeLast', 'TLBalHCPct', 'TLSatPct', 'TLOpenPct', 'TLOpen24Pct'
]

GROUP_3_SUMS_MAXSUMS = [
    'TLSum', 'TLMaxSum'
]


# Dummy target column name for model training
DUMMY_TARGET_COLUMN_NAME = 'TARGET' 

# --- Simulate Model Training and Preprocessing ---
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
        'TLDel60Cnt24': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Added to dummy data
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
    pd.DataFrame({ # Bad applicants (TARGET = 1) - high derogatory, low credit age, high inquiries
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
        'TLDel60Cnt24': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1], # Added to dummy data
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

    # --- IMPORTANT: Handle duplicate column name from CSV ---
    # Pandas renames duplicate columns with '.1', '.2', etc.
    # We assume the first instance of 'TLDel60Cnt24' is the one we want to use for the model.
    # If 'TLDel60Cnt24.1' exists, it means there was a duplicate in the CSV. Drop it.
    if 'TLDel60Cnt24.1' in df_processed.columns:
        st.write("DEBUG: Dropping duplicate column 'TLDel60Cnt24.1' found in uploaded CSV.")
        df_processed = df_processed.drop(columns=['TLDel60Cnt24.1'])


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
    st.write(f"DEBUG: Columns used for prediction: {df_processed_for_prediction.columns.tolist()}")


    probabilities = model_pipeline.predict_proba(df_processed_for_prediction)
    
    # Assuming class 0 (index 0) is "Good" and class 1 (index 1) is "Bad" (Default)
    # So, score is probability of NOT defaulting (Class 0)
    scores = probabilities[:, 0] * 100 

    decisions = np.where(scores >= approval_threshold, "Approved", "Rejected") 

    return pd.DataFrame({'Score': scores, 'Decision': decisions})


# --- Streamlit App Layout ---
st.set_page_config(page_title="Credit Scoring Dashboard", layout="wide")

st.title("Credit Scoring Dashboard")

# --- Sidebar for Data Management and Analysis ---
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
        # Clear previous analysis data to force re-analysis
        st.session_state['scored_data'] = pd.DataFrame()
        # No need to set analyze_triggered here, as analysis is now direct
        st.sidebar.info("File uploaded. Please select it from the dropdown below and click 'Analyze'.")
        time.sleep(1) # Give user a moment to read success message
        clear_file_uploader() # Reset uploader widget
        st.rerun() # Rerun to update the selectbox options

    except Exception as e:
        st.sidebar.error(f"Error uploading file to S3: {e}")

st.sidebar.subheader("Analyze Credit Data from S3")

s3_files = []
try:
    if s3_client:
        response = s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix="uploads/")
        if 'Contents' in response:
            s3_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.csv')]
            s3_files.sort(reverse=True) # Show most recent first
except Exception as e:
    st.sidebar.warning(f"Could not list files from S3 bucket: {e}. Ensure 'uploads/' prefix exists or bucket is not empty, and permissions are correct.")
    s3_files = []

# --- Analysis Trigger Form ---
with st.sidebar.form("analysis_trigger_form"):
    st.markdown("### Select File for Analysis")

    selected_s3_file_in_form = None
    if s3_files:
        default_index = 0
        if st.session_state['last_uploaded_s3_key'] and st.session_state['last_uploaded_s3_key'] in s3_files:
            try:
                default_index = s3_files.index(st.session_state['last_uploaded_s3_key'])
            except ValueError: # Fallback if key is not found (e.g., deleted)
                default_index = 0

        selected_s3_file_in_form = st.selectbox(
            "Choose a CSV file from S3:",
            options=s3_files,
            index=default_index,
            key="s3_file_selector_form" # Unique key for the selectbox
        )
    else:
        st.info("No CSV files found in the 'uploads/' folder of your S3 bucket. Please upload one above.")

    if selected_s3_file_in_form:
        try:
            # Temporarily read the file to get column names for target selector
            temp_obj = s3_client.get_object(Bucket=s3_bucket_name, Key=selected_s3_file_in_form)
            temp_df = pd.read_csv(io.BytesIO(temp_obj['Body'].read()))
            st.session_state['uploaded_df_columns'] = temp_df.columns.tolist()

            default_target_idx = 0
            # Prioritize 'TARGET' as it was the original default for credit.csv
            if 'TARGET' in st.session_state['uploaded_df_columns']:
                default_target_idx = st.session_state['uploaded_df_columns'].index('TARGET')
            elif 'default' in st.session_state['uploaded_df_columns']: # Also check for 'default'
                default_target_idx = st.session_state['uploaded_df_columns'].index('default')
            elif len(st.session_state['uploaded_df_columns']) > 0:
                st.sidebar.warning(f"Default target column 'TARGET' or 'default' not found. Please select the correct target column.")
                default_target_idx = 0 # Default to first column if common targets not found

            st.session_state['selected_target_column'] = st.selectbox(
                "Select the Target Column (e.g., 'default', 'TARGET'):",
                options=st.session_state['uploaded_df_columns'],
                index=default_target_idx,
                key="target_column_selector"
            )
        except Exception as e:
            st.sidebar.error(f"Error reading selected file to get columns for target selector: {e}")
            st.session_state['uploaded_df_columns'] = [] # Clear columns on error
            st.session_state['selected_target_column'] = '' # Clear selected target on error
    else:
        st.session_state['uploaded_df_columns'] = []
        st.session_state['selected_target_column'] = ''


    analyze_submitted = st.form_submit_button(f"Analyze Selected File")

    if analyze_submitted:
        if selected_s3_file_in_form and st.session_state['selected_target_column']:
            file_to_analyze = selected_s3_file_in_form
            actual_target_col_name = st.session_state['selected_target_column']

            if not actual_target_col_name:
                st.error("No target column selected. Please select one from the dropdown.")
            else:
                try:
                    with st.spinner(f"Downloading and analyzing '{file_to_analyze}' from S3..."):
                        obj = s3_client.get_object(Bucket=s3_bucket_name, Key=file_to_analyze)
                        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
                        st.write(f"DEBUG: Original columns in loaded CSV: {df.columns.tolist()}")

                        # --- CRITICAL FIX: Handle duplicate column name from CSV ---
                        # Pandas renames duplicate columns with '.1', '.2', etc.
                        # We assume the first instance of 'TLDel60Cnt24' is the one we want to use for the model.
                        # If 'TLDel60Cnt24.1' exists, it means there was a duplicate in the CSV. Drop it.
                        if 'TLDel60Cnt24.1' in df.columns:
                            st.write("DEBUG: Dropping duplicate column 'TLDel60Cnt24.1' found in uploaded CSV.")
                            df = df.drop(columns=['TLDel60Cnt24.1'])


                        # Ensure 'ID' column exists for merging later
                        if 'ID' not in df.columns:
                            df['ID'] = df.index.astype(str) + '_idx'
                        else:
                            df['ID'] = df['ID'].astype(str) # Ensure ID is string type

                        # Convert target column to numeric and handle NaNs
                        if actual_target_col_name in df.columns:
                            df[actual_target_col_name] = pd.to_numeric(df[actual_target_col_name], errors='coerce').fillna(0).astype(int)
                        else:
                            st.error(f"The selected target column '{actual_target_col_name}' was not found in the uploaded data. Please check your CSV and select the correct column.")
                            st.session_state['scored_data'] = pd.DataFrame() # Clear on error
                            return # Exit function early if target column is missing

                        st.write(f"DEBUG: DataFrame columns before passing to ML scoring: {df.columns.tolist()}")
                        results_df = get_credit_score_ml(df.copy(), st.session_state['approval_threshold'], actual_target_col_name)
                        st.write(f"DEBUG: Results DataFrame from ML scoring: {results_df.head()}")

                        # Reorder columns to have ID first, then original columns, then Score and Decision
                        if 'ID' in df.columns:
                            cols = ['ID'] + [col for col in df.columns if col != 'ID']
                            df = df[cols]

                        df_scored = pd.concat([df, results_df], axis=1)
                        st.session_state['scored_data'] = df_scored
                        st.success(f"Analysis complete for '{file_to_analyze}'.")
                        st.write(f"DEBUG: Final scored_data shape after analysis: {st.session_state['scored_data'].shape}")
                        st.rerun() # Force a rerun to display the dashboard with new data

                except Exception as e:
                    st.error(f"Error analyzing file from S3: {e}. Please check file format and column names in your CSV file.")
                    st.write(f"Detailed error: {e}") # Provide more detailed error to user
                    st.session_state['scored_data'] = pd.DataFrame() # Clear on error
        else:
            st.sidebar.warning("Please select a file and a target column to analyze.")

# --- Clear Displayed Data Button (in sidebar) ---
# This button should only appear if there is data to clear
if not st.session_state['scored_data'].empty:
    if st.sidebar.button("Clear Displayed Data", key="clear_data_button_sidebar"):
        st.session_state['scored_data'] = pd.DataFrame()
        st.session_state['selected_file_for_analysis'] = None
        # No need to reset analyze_triggered as it's not used in the new flow
        st.sidebar.success("Displayed data cleared.")
        st.rerun()


# --- Display Dashboard Results if scored_data is available ---
if not st.session_state['scored_data'].empty:
    df_display = st.session_state['scored_data']
    st.write("DEBUG: Displaying dashboard because scored_data is not empty.")

    st.header("Credit Scoring Dashboard")

    # Display overall metrics
    col1, col2, col3 = st.columns(3)

    total_applicants = len(df_display)
    approved_applicants = df_display[df_display['Decision'] == 'Approved'].shape[0]
    rejected_applicants = df_display[df_display['Decision'] == 'Rejected'].shape[0]
    approval_rate = (approved_applicants / total_applicants * 100) if total_applicants > 0 else 0

    with col1:
        st.metric("Total Applicants", total_applicants)
    with col2:
        st.metric("Approved Applicants", approved_applicants)
    with col3:
        st.metric("Approval Rate", f"{approval_rate:.2f}%")

    st.markdown("---")

    # Score Distribution Histogram
    st.subheader("Credit Score Distribution")
    fig_hist = px.histogram(df_display, x="Score", nbins=20,
                            title="Distribution of Credit Scores",
                            labels={"Score": "Credit Score"},
                            color_discrete_sequence=['skyblue'])
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # Decision Breakdown Pie Chart
    st.subheader("Loan Decision Breakdown")
    decision_counts = df_display['Decision'].value_counts().reset_index()
    decision_counts.columns = ['Decision', 'Count']
    fig_pie = px.pie(decision_counts, values='Count', names='Decision',
                     title='Proportion of Approved vs. Rejected Loans',
                     color_discrete_sequence=['lightgreen', 'salmon'])
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # Average feature values by decision - Group 1
    st.subheader("Average Statistics by Loan Decision (Group 1: Counts & Indicators)")
    features_group1 = [col for col in GROUP_1_COUNTS_INDICATORS if col in df_display.columns]
    if features_group1:
        # Ensure numerical columns are truly numeric before aggregation
        for col in features_group1:
            df_display[col] = pd.to_numeric(df_display[col], errors='coerce').fillna(0)
        
        avg_features_by_decision_group1 = df_display.groupby('Decision')[features_group1].mean().T
        st.dataframe(avg_features_by_decision_group1)
    else:
        st.info("No features in Group 1 available for showing average values by decision.")

    st.markdown("---")

    # Average feature values by decision - Group 2
    st.subheader("Average Statistics by Loan Decision (Group 2: Time & Percentages)")
    features_group2 = [col for col in GROUP_2_TIME_PERCENTAGES if col in df_display.columns]
    if features_group2:
        # Ensure numerical columns are truly numeric before aggregation
        for col in features_group2:
            df_display[col] = pd.to_numeric(df_display[col], errors='coerce').fillna(0)

        avg_features_by_decision_group2 = df_display.groupby('Decision')[features_group2].mean().T
        st.dataframe(avg_features_by_decision_group2)
    else:
        st.info("No features in Group 2 available for showing average values by decision.")

    st.markdown("---")

    # Average feature values by decision - Group 3
    st.subheader("Average Statistics by Loan Decision (Group 3: Sums & Max Sums)")
    features_group3 = [col for col in GROUP_3_SUMS_MAXSUMS if col in df_display.columns]
    if features_group3:
        # Ensure numerical columns are truly numeric before aggregation
        for col in features_group3:
            df_display[col] = pd.to_numeric(df_display[col], errors='coerce').fillna(0)

        avg_features_by_decision_group3 = df_display.groupby('Decision')[features_group3].mean().T
        st.dataframe(avg_features_by_decision_group3)
    else:
        st.info("No features in Group 3 available for showing average values by decision.")

    st.markdown("---")

    # Display the scored data table
    st.subheader("Scored Credit Data")
    st.dataframe(df_display)

else:
    st.info("Upload a CSV file and click 'Analyze' to view the dashboard.")
    st.write("DEBUG: scored_data is empty, so dashboard is not displayed.")


st.markdown("---")
st.subheader("How this app would integrate with AWS and LLM (Recap):")

st.markdown(
    """
    This Streamlit app demonstrates uploading and analyzing **sophisticated credit data** from AWS S3, leveraging a powerful Neural Network machine learning model.

    ### ☁️ AWS Cloud Integration:
    * **Data Storage (S3):** Data files are stored in AWS S3, providing durable and scalable storage.
    * **Machine Learning Model Hosting (SageMaker):** In a real scenario, the `get_credit_score_ml` function would call a sophisticated ML model deployed on AWS SageMaker for more accurate and robust predictions. This separates the heavy computation from the Streamlit app.
    * **Serverless Functions (Lambda):** Could be used for automated processing of new files uploaded to S3 (e.g., triggering the scoring process automatically).
    * **Authentication & Authorization (Cognito):** For secure user access to the app and S3, ensuring only authorized users can upload or view sensitive data.
    * **Logging & Monitoring (CloudWatch):** To track app performance, S3 interactions, and potential errors, providing insights for operational management.

    ### 🧠 Large Language Model (LLM) Integration (Conceptual):
    While not directly implemented in this version, an LLM could be used to provide **Explainable AI (XAI)**. After identifying rejected applicants, an LLM could generate concise, human-readable explanations for *why* specific applicants were flagged as high-risk, based on their input features. This helps in understanding and communicating complex model decisions.
    """
)
