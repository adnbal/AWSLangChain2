import streamlit as st
import pandas as pd
import numpy as np
import boto3
import io
import datetime # For unique file naming
import os
import time # For a small delay

# --- Dummy Credit Scoring Model ---
# This function applies the scoring logic to a single row of data.
# In a real application, this would be a call to a deployed ML model (e.g., AWS SageMaker endpoint).
def get_credit_score(data_row):
    """
    A simple dummy credit scoring function.
    It takes a dictionary-like object (representing a row of data) and returns a score and a decision.
    """
    # Define mappings for categorical variables
    reason_map = {'DebtCon': 0.3, 'HomeImp': 0.2, 'Other': 0.1}
    job_map = {'Mgr': 0.2, 'Office': 0.1, 'Other': 0.05, 'ProfExe': 0.15, 'Sales': 0.1, 'Self': 0.25}

    # Safely get numerical inputs, providing default 0 if a column is missing
    # This makes the function more robust to variations in uploaded Excel files.
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

    # Safely get categorical inputs, providing 'Other' as default if missing
    reason_score = reason_map.get(data_row.get('REASON', 'Other'), 0.1)
    job_score = job_map.get(data_row.get('JOB', 'Other'), 0.05)

    # Simple heuristic for a dummy score (higher is better)
    # This calculation is for demonstration purposes only and is not a real credit model.
    score = (
        (loan / 10000) * 5 + # Larger loans might imply higher trust or risk depending on context
        (value / 100000) * 10 - # Higher value of property is good
        (mortdue / 100000) * 5 + # Higher mortgage due is bad
        yojs * 0.5 - # Years on job, more is better
        derog * 20 - # Derogatory reports, more is bad
        delinq * 15 - # Delinquent credit lines, more is bad
        (clage / 100) * 2 + # Age of oldest credit line, more is better
        ninq * 10 - # Number of recent credit inquiries, more is bad
        clno * 1 + # Number of credit lines, more is generally good
        debtinc * 30 + # Debt-to-income ratio, higher is bad
        reason_score * 50 +
        job_score * 50
    )

    # Normalize score to a 0-100 range and clamp it
    score = max(0, min(100, score))

    # Determine decision based on score threshold
    decision = "Approved" if score >= 60 else "Rejected"
    
    # Return results as a Pandas Series, which is convenient for .apply()
    return pd.Series({'Score': score, 'Decision': decision})

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

st.title("Credit Scoring Dashboard with S3 Integration")
st.subheader("Upload Excel files and analyze loan vulnerability")

# --- Initialize S3 Client ---
s3_client = None
s3_bucket_name = None
aws_region_name = None

try:
    # --- Accessing secrets as nested keys, matching your Streamlit Cloud config ---
    # Based on your screenshot, your secrets are defined with [aws] section.
    aws_access_key = st.secrets["aws"]["access_key_id"]
    aws_secret_key = st.secrets["aws"]["secret_access_key"]
    s3_bucket_name = st.secrets["aws"]["s3_bucket_name"]
    aws_region_name = st.secrets["aws"]["region_name"]
    # If you had other top-level secrets (like OPENAI_API_KEY), you'd access them like:
    # openai_api_key = st.secrets.get("OPENAI_API_KEY", "NOT_SET") 

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
    # if 'openai_api_key' in locals() and openai_api_key != "NOT_SET":
    #    st.sidebar.write(f"OpenAI API Key: `{openai_api_key[:4]}...`")

except KeyError as e:
    st.sidebar.error(f"Secret key not found: {e}. Please ensure your secrets are configured correctly as nested keys under `[aws]` in Streamlit Cloud or `.streamlit/secrets.toml`.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"An error occurred while initializing S3 client: {e}")
    st.stop()

# --- File Upload Section ---
st.header("Upload Excel File to S3")
uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    s3_file_key = f"uploads/{timestamp}_{file_name}"

    try:
        with st.spinner(f"Uploading {file_name} to S3..."):
            s3_client.upload_fileobj(uploaded_file, s3_bucket_name, s3_file_key)
        st.success(f"File '{file_name}' uploaded successfully to S3 as '{s3_file_key}'!")
        st.session_state['last_uploaded_s3_key'] = s3_file_key
        # --- IMPORTANT: REMOVED st.rerun() / st.experimental_rerun() ---
        # This line was causing the loop due to older Streamlit versions.
        # User will manually select the uploaded file from the dropdown.
        st.info("File uploaded. Please select it from the dropdown below and click 'Analyze'.")
        time.sleep(1) # A small delay to ensure the info message is seen
    except Exception as e:
        st.error(f"Error uploading file to S3: {e}")

# --- Dashboard Section ---
st.header("Analyze Loans from S3")

s3_files = []
try:
    if s3_client:
        response = s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix="uploads/")
        if 'Contents' in response:
            s3_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.xlsx')]
            s3_files.sort(reverse=True)
except Exception as e:
    st.warning(f"Could not list files from S3 bucket: {e}. Ensure 'uploads/' prefix exists or bucket is not empty, and permissions are correct.")
    s3_files = []

selected_s3_file = None
if s3_files:
    default_index = 0
    if 'last_uploaded_s3_key' in st.session_state and st.session_state['last_uploaded_s3_key'] in s3_files:
        try:
            default_index = s3_files.index(st.session_state['last_uploaded_s3_key'])
        except ValueError:
            default_index = 0

    selected_s3_file = st.selectbox(
        "Select an Excel file from S3 to analyze:",
        options=s3_files,
        index=default_index,
        key="s3_file_selector"
    )
else:
    st.info("No Excel files found in the 'uploads/' folder of your S3 bucket. Please upload one above.")

# --- Analyze Button and Logic (Directly) ---
st.markdown("### Loan Analysis")

if selected_s3_file:
    if st.button(f"Analyze '{selected_s3_file}'", key="analyze_button"):
        st.write("DEBUG: Analyze button was clicked. Starting analysis...")
        try:
            with st.spinner(f"Downloading and analyzing '{selected_s3_file}' from S3..."):
                obj = s3_client.get_object(Bucket=s3_bucket_name, Key=selected_s3_file)
                excel_data = obj['Body'].read()

                df = pd.read_excel(io.BytesIO(excel_data))
                st.write(f"DEBUG: Successfully read {len(df)} rows from Excel file.")
                st.write("DEBUG: Columns in loaded Excel file:", df.columns.tolist()) # CRITICAL DEBUGGING POINT

                results = df.apply(get_credit_score, axis=1)
                df_scored = pd.concat([df, results], axis=1)
                st.write(f"DEBUG: Scored data has {len(df_scored)} rows.")

                st.session_state['scored_data'] = df_scored
                st.success(f"Analysis complete for '{selected_s3_file}'.")

        except Exception as e:
            st.error(f"Error analyzing file from S3: {e}. Please check file format and column names in your Excel file.")
            st.write("DEBUG: An error occurred during analysis:", e)
            st.session_state['scored_data'] = pd.DataFrame()
    else:
        st.write("DEBUG: Analyze button not yet clicked or state reset.")
else:
    st.info("Please select an Excel file from the dropdown to enable analysis.")


# --- Display Dashboard Results ---
st.markdown("### Dashboard Display")
if 'scored_data' in st.session_state and not st.session_state['scored_data'].empty:
    df_display = st.session_state['scored_data']
    st.write("DEBUG: Displaying scored data.")

    st.subheader("Full Loan Data with Scores and Decisions")
    st.dataframe(df_display)

    st.subheader("Vulnerable Loans (Rejected)")
    vulnerable_loans = df_display[df_display['Decision'] == 'Rejected']

    if not vulnerable_loans.empty:
        st.dataframe(vulnerable_loans)
        st.info(f"Found {len(vulnerable_loans)} vulnerable loans.")
    else:
        st.success("No vulnerable loans found in this dataset!")
else:
    st.write("DEBUG: No scored data found in session state or data is empty. Dashboard not displayed.")

# --- AWS and Langchain Integration Explanation (Recap) ---
st.markdown("---")
st.subheader("How this app would integrate with AWS and Langchain (Recap):")

st.markdown(
    """
    This Streamlit app now demonstrates uploading and analyzing Excel files from AWS S3.

    ### ☁️ AWS Cloud Integration:
    * **Data Storage (S3):** Excel files are now stored in AWS S3, providing durable and scalable storage.
    * **Machine Learning Model Hosting (SageMaker):** In a real scenario, the `get_credit_score` function would call a sophisticated ML model deployed on AWS SageMaker for more accurate and robust predictions. This separates the heavy computation from the Streamlit app.
    * **Serverless Functions (Lambda):** Could be used for automated processing of new files uploaded to S3 (e.g., triggering the scoring process automatically).
    * **Authentication & Authorization (Cognito):** For secure user access to the app and S3, ensuring only authorized users can upload or view sensitive loan data.
    * **Logging & Monitoring (CloudWatch):** To track app performance, S3 interactions, and potential errors, providing insights for operational management.

    ### 🔗 Langchain Integration:
    Langchain is primarily used for building applications with Large Language Models (LLMs). It could enhance this application in several ways:
    * **Explainable AI (XAI):** After identifying vulnerable loans, Langchain could prompt an LLM to generate more detailed, human-readable explanations for *why* specific loans were flagged as vulnerable, based on their input features.
    * **Conversational Interface:** Users could interact with the dashboard using natural language queries (e.g., "Show me all rejected loans with a debt-to-income ratio above 40%"), with Langchain interpreting the query and dynamically filtering the DataFrame.
    * **Automated Reporting:** Langchain could help generate summary reports or alerts for vulnerable loans, potentially integrating with email services to notify relevant stakeholders.
    """
)
