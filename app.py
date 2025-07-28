import streamlit as st
import pandas as pd
import numpy as np
import boto3
import io
import datetime # For unique file naming
import os

# --- Dummy Credit Scoring Model ---
# This function remains the same, but will now operate on DataFrame rows.
def get_credit_score(data_row):
    """
    A simple dummy credit scoring function.
    It takes a dictionary (representing a row of data) and returns a score and a decision.
    """
    reason_map = {'DebtCon': 0.3, 'HomeImp': 0.2, 'Other': 0.1}
    job_map = {'Mgr': 0.2, 'Office': 0.1, 'Other': 0.05, 'ProfExe': 0.15, 'Sales': 0.1, 'Self': 0.25}

    # Use .get() with a default value to handle potential missing columns in uploaded data
    loan = data_row.get('LOAN', 0)
    mortdue = data_row.get('MORTDUE', 0)
    value = data_row.get('VALUE', 0)
    yojs = data_row.get('YOJ', 0)
    derog = data_row.get('DEROG', 0)
    delinq = data_row.get('DELINQ', 0)
    clage = data_row.get('CLAGE', 0)
    ninq = data_row.get('NINQ', 0)
    clno = data_row.get('CLNO', 0)
    debtinc = data_row.get('DEBTINC', 0)

    reason_score = reason_map.get(data_row.get('REASON', 'Other'), 0.1)
    job_score = job_map.get(data_row.get('JOB', 'Other'), 0.05)

    score = (
        (loan / 10000) * 5 +
        (value / 100000) * 10 -
        (mortdue / 100000) * 5 +
        yojs * 0.5 -
        derog * 20 -
        delinq * 15 -
        (clage / 100) * 2 +
        ninq * 10 -
        clno * 1 +
        debtinc * 30 +
        reason_score * 50 +
        job_score * 50
    )

    score = max(0, min(100, score))
    decision = "Approved" if score >= 60 else "Rejected"
    return pd.Series({'Score': score, 'Decision': decision})

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Dynamic Credit Scoring Dashboard", layout="wide")

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

st.title("Credit Scoring Dashboard with S3 Integration")
st.subheader("Upload Excel files and analyze loan vulnerability")

# --- Initialize S3 Client ---
s3_client = None
s3_bucket_name = None

try:
    aws_access_key = st.secrets["aws"]["access_key_id"]
    aws_secret_key = st.secrets["aws"]["secret_access_key"]
    s3_bucket_name = st.secrets["aws"]["s3_bucket_name"]

    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )
    st.sidebar.success("AWS S3 client initialized!")

    # Display secrets (partially masked) in sidebar for verification
    st.sidebar.header("Application Secrets")
    st.sidebar.write(f"AWS Access Key ID: `{aws_access_key[:4]}...`")
    st.sidebar.write(f"AWS Secret Access Key: `{aws_secret_key[:4]}...`")
    st.sidebar.write(f"S3 Bucket Name: `{s3_bucket_name}`")

except KeyError as e:
    st.sidebar.error(f"Secret key not found: {e}. Please ensure your `.streamlit/secrets.toml` is configured correctly or secrets are set in Streamlit Cloud.")
    st.stop() # Stop execution if essential secrets are missing
except Exception as e:
    st.sidebar.error(f"An error occurred while initializing S3 client: {e}")
    st.stop()

# --- File Upload Section ---
st.header("Upload Excel File to S3")
uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    # Create a unique file name in S3 to avoid overwrites
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    s3_file_key = f"uploads/{timestamp}_{file_name}"

    try:
        # Upload the file to S3
        with st.spinner(f"Uploading {file_name} to S3..."):
            s3_client.upload_fileobj(uploaded_file, s3_bucket_name, s3_file_key)
        st.success(f"File '{file_name}' uploaded successfully to S3 as '{s3_file_key}'!")
        st.session_state['last_uploaded_s3_key'] = s3_file_key # Store for later retrieval
    except Exception as e:
        st.error(f"Error uploading file to S3: {e}")

# --- Dashboard Section ---
st.header("Analyze Loans from S3")

# Option to select a file from S3 (or use the last uploaded one)
s3_files = []
try:
    if s3_client:
        response = s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix="uploads/")
        if 'Contents' in response:
            s3_files = [obj['Key'] for obj in response['Contents']]
            s3_files.sort(reverse=True) # Show most recent first
except Exception as e:
    st.warning(f"Could not list files from S3 bucket: {e}. Ensure 'uploads/' prefix exists or bucket is not empty.")
    s3_files = [] # Reset to empty list if error

selected_s3_file = None
if s3_files:
    # Pre-select the last uploaded file if available
    default_index = 0
    if 'last_uploaded_s3_key' in st.session_state and st.session_state['last_uploaded_s3_key'] in s3_files:
        default_index = s3_files.index(st.session_state['last_uploaded_s3_key'])

    selected_s3_file = st.selectbox(
        "Select an Excel file from S3 to analyze:",
        options=s3_files,
        index=default_index
    )
else:
    st.info("No Excel files found in the 'uploads/' folder of your S3 bucket. Please upload one above.")


if selected_s3_file and st.button(f"Analyze '{selected_s3_file}'"):
    try:
        with st.spinner(f"Downloading and analyzing '{selected_s3_file}' from S3..."):
            # Download file from S3
            obj = s3_client.get_object(Bucket=s3_bucket_name, Key=selected_s3_file)
            excel_data = obj['Body'].read()

            # Read Excel data into pandas DataFrame
            df = pd.read_excel(io.BytesIO(excel_data))

            # Apply credit scoring to each row
            # Use .apply() with axis=1 to pass each row as a dictionary-like object
            results = df.apply(get_credit_score, axis=1)
            df_scored = pd.concat([df, results], axis=1)

            st.session_state['scored_data'] = df_scored # Store for filtering

            st.success(f"Analysis complete for '{selected_s3_file}'.")

    except Exception as e:
        st.error(f"Error analyzing file from S3: {e}")
        st.session_state['scored_data'] = pd.DataFrame() # Clear data on error

# --- Display Dashboard ---
if 'scored_data' in st.session_state and not st.session_state['scored_data'].empty:
    df_display = st.session_state['scored_data']

    st.subheader("Full Loan Data with Scores and Decisions")
    st.dataframe(df_display)

    st.subheader("Vulnerable Loans (Rejected)")
    vulnerable_loans = df_display[df_display['Decision'] == 'Rejected']

    if not vulnerable_loans.empty:
        st.dataframe(vulnerable_loans)
        st.info(f"Found {len(vulnerable_loans)} vulnerable loans.")
    else:
        st.success("No vulnerable loans found in this dataset!")

# --- AWS and Langchain Integration Explanation (remains the same) ---
st.markdown("---")
st.subheader("How this app would integrate with AWS and Langchain (Recap):")

st.markdown(
    """
    This Streamlit app now demonstrates uploading and analyzing Excel files from AWS S3.

    ### ☁️ AWS Cloud Integration:
    * **Data Storage (S3):** Excel files are now stored in AWS S3, providing durable and scalable storage.
    * **Machine Learning Model Hosting (SageMaker):** In a real scenario, the `get_credit_score` function would call a model deployed on AWS SageMaker for more accurate predictions.
    * **Serverless Functions (Lambda):** Could be used for automated processing of new files uploaded to S3.
    * **Authentication & Authorization (Cognito):** For secure user access to the app and S3.
    * **Logging & Monitoring (CloudWatch):** To track app performance and S3 interactions.

    ### 🔗 Langchain Integration:
    Langchain is primarily used for building applications with Large Language Models (LLMs). It could enhance this application in several ways:
    * **Explainable AI (XAI):** After identifying vulnerable loans, Langchain could prompt an LLM to generate more detailed explanations for why certain loans were flagged, based on their specific features.
    * **Conversational Interface:** Users could query the dashboard using natural language (e.g., "Show me all rejected loans with high debt-to-income ratio"), with Langchain interpreting the query and filtering the DataFrame.
    * **Automated Reporting:** Langchain could help generate summary reports or alerts for vulnerable loans, potentially integrating with email services.

    ### 🚀 GitHub & Streamlit Cloud Deployment:
    1.  **GitHub Repository:** Your repository should contain `app.py` and `requirements.txt`. **Crucially, `.streamlit/secrets.toml` should NOT be committed.**
    2.  **Streamlit Cloud:** Deploy from GitHub. Remember to add your AWS `access_key_id`, `secret_access_key`, and `s3_bucket_name` as secrets in the Streamlit Cloud dashboard (e.g., `aws.access_key_id`, `aws.secret_access_key`, `aws.s3_bucket_name`).
    """
)
