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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# --- Initialize session state variables using setdefault for robustness ---
# This ensures these keys always exist with a default value if not already set.
st.session_state.setdefault('scored_data', pd.DataFrame())
st.session_state.setdefault('analyze_triggered', False)
st.session_state.setdefault('last_uploaded_s3_key', None)
st.session_state.setdefault('selected_file_for_analysis', None)
st.session_state.setdefault('file_uploader_key', 0) # Initialize key for file uploader

# --- Define Feature Columns (based on your homeequity.xlsx) ---
NUMERICAL_FEATURES = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']
CATEGORICAL_FEATURES = ['REASON', 'JOB']
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# --- Simulate Model Training and Preprocessing ---
# In a real scenario, this would be done offline and saved/loaded.
# We create a dummy dataset to fit the preprocessor and model.
# This dummy data needs to contain examples of all possible categorical values
# that the real data might have, to ensure the OneHotEncoder is fitted correctly.
dummy_data_for_training = pd.DataFrame({
    'LOAN': [10000, 20000, 5000, 15000, 8000],
    'MORTDUE': [0, 50000, 0, 20000, 10000],
    'VALUE': [50000, 100000, 30000, 75000, 60000],
    'REASON': ['DebtCon', 'HomeImp', 'Other', 'DebtCon', 'HomeImp'],
    'JOB': ['Mgr', 'ProfExe', 'Office', 'Sales', 'Self'],
    'YOJ': [5, 10, 2, 7, 15],
    'DEROG': [0, 1, 0, 0, 2],
    'DELINQ': [0, 0, 1, 0, 0],
    'CLAGE': [100, 200, 50, 150, 120],
    'NINQ': [0, 1, 0, 2, 0],
    'CLNO': [10, 15, 5, 12, 8],
    'DEBTINC': [35.0, 40.0, 25.0, 30.0, 45.0],
    'BAD': [0, 1, 0, 1, 0] # Dummy target variable
})

# Define preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERICAL_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES) # handle_unknown='ignore' is crucial for new categories
    ])

# Create a pipeline: preprocess then apply logistic regression
# In a real scenario, the model would be trained on the 'BAD' column
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')) # class_weight helps with imbalanced data
])

# "Fit" the pipeline on the dummy data
# This simulates loading a pre-trained model and preprocessor
try:
    model_pipeline.fit(dummy_data_for_training[ALL_FEATURES], dummy_data_for_training['BAD'])
    st.sidebar.success("Simulated ML model and preprocessor initialized.")
except Exception as e:
    st.sidebar.error(f"Error initializing simulated ML model: {e}")
    st.stop() # Stop if model can't be initialized

# --- Credit Scoring Function using the ML Model ---
@st.cache_data # Cache the results of this function for performance
def get_credit_score_ml(df_input: pd.DataFrame):
    """
    Applies the simulated ML model to score a DataFrame of loan applications.
    """
    # Ensure input DataFrame has all expected features, fill missing with 0 or 'Other'
    # This is important for the preprocessor to work correctly
    for col in NUMERICAL_FEATURES:
        if col not in df_input.columns:
            df_input[col] = 0
    for col in CATEGORICAL_FEATURES:
        if col not in df_input.columns:
            df_input[col] = 'Other' # Default for missing categorical

    # Select and reorder columns to match the training order
    df_processed_for_prediction = df_input[ALL_FEATURES]

    # Predict probabilities (probability of BAD=1, BAD=0)
    # We want the probability of being a "good" loan (BAD=0) or "bad" loan (BAD=1)
    # LogisticRegression.predict_proba returns [[prob_class_0, prob_class_1], ...]
    # We'll use prob_class_0 for "good" score, higher is better.
    probabilities = model_pipeline.predict_proba(df_processed_for_prediction)
    
    # Assuming class 0 is "Good" and class 1 is "Bad"
    # If your 'BAD' column is 0 for good, 1 for bad, then probabilities[:, 0] is P(Good)
    scores = probabilities[:, 0] * 100 # Scale to 0-100

    # Make decisions based on a threshold (can be tuned)
    decisions = np.where(scores >= 50, "Approved", "Rejected") # Using 50 as a default threshold for probability score

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

st.title("Credit Scoring Dashboard with S3 Integration (ML Powered)")
st.subheader("Upload Excel files and analyze loan vulnerability")

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
    # Increment the key to force Streamlit to re-render the uploader widget, effectively clearing it
    st.session_state['file_uploader_key'] += 1
    # Note: We don't set the widget's value to None directly via session_state here,
    # as incrementing the key is the recommended way to clear it.

# --- File Upload Section ---
st.header("Upload Excel File to S3")
uploaded_file = st.file_uploader(
    "Choose an Excel file (.xlsx)",
    type=["xlsx"],
    key=f"file_uploader_{st.session_state['file_uploader_key']}" # Dynamic key for clearing
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
        st.session_state['analyze_triggered'] = False # Reset trigger after upload
        st.session_state['scored_data'] = pd.DataFrame() # Clear old data
        st.info("File uploaded. Please select it from the dropdown below and click 'Analyze'.")
        time.sleep(1) # Small delay for message visibility

        # --- CRITICAL FIX: Clear the uploader state and rerun ---
        clear_file_uploader() # Clear the widget's internal state
        st.rerun() # Force a rerun to reflect the cleared uploader and updated file list

    except Exception as e:
        st.error(f"Error uploading file to S3: {e}")

# --- Dashboard Section for Analysis Trigger (Encapsulated in a Form) ---
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
            "Choose an Excel file from S3:",
            options=s3_files,
            index=default_index,
            key="s3_file_selector_form" # Unique key for selectbox within form
        )
    else:
        st.info("No Excel files found in the 'uploads/' folder of your S3 bucket. Please upload one above.")

    # This is the button that submits the form and triggers analysis
    analyze_submitted = st.form_submit_button(f"Analyze Selected File")

    if analyze_submitted:
        if selected_s3_file_in_form:
            st.write(f"DEBUG: Analyze form submitted. Selected file: `{selected_s3_file_in_form}`. Starting analysis...")
            st.session_state['selected_file_for_analysis'] = selected_s3_file_in_form # Store the selected file
            st.session_state['analyze_triggered'] = True # Set the trigger
            st.rerun() # Rerun to process the analysis trigger
        else:
            st.warning("Please select a file to analyze.")
            st.session_state['analyze_triggered'] = False
    else:
        st.write("DEBUG: Analysis form not yet submitted.")

# --- Perform Analysis if Triggered ---
# This block runs on a subsequent rerun after the form is submitted
if st.session_state['analyze_triggered'] and st.session_state['selected_file_for_analysis']:
    file_to_analyze = st.session_state['selected_file_for_analysis']
    st.write(f"DEBUG: Analysis triggered via session state for file: `{file_to_analyze}`.")
    try:
        with st.spinner(f"Downloading and analyzing '{file_to_analyze}' from S3..."):
            obj = s3_client.get_object(Bucket=s3_bucket_name, Key=file_to_analyze)
            excel_data = obj['Body'].read()

            df = pd.read_excel(io.BytesIO(excel_data))
            st.write(f"DEBUG: Successfully read {len(df)} rows from Excel file.")
            st.write("DEBUG: Columns in loaded Excel file:", df.columns.tolist()) # CRITICAL DEBUGGING POINT

            # --- CHANGED: Call the ML-powered scoring function ---
            results_df = get_credit_score_ml(df.copy()) # Pass a copy to avoid modifying original df
            df_scored = pd.concat([df, results_df], axis=1)
            st.write(f"DEBUG: Scored data has {len(df_scored)} rows.")

            st.session_state['scored_data'] = df_scored
            st.success(f"Analysis complete for '{file_to_analyze}'.")
            st.session_state['analyze_triggered'] = False # Reset trigger after analysis
            st.session_state['selected_file_for_analysis'] = None # Clear selected file after analysis
            st.write("DEBUG: Analysis complete. Trigger and selected file reset.")

    except Exception as e:
        st.error(f"Error analyzing file from S3: {e}. Please check file format and column names in your Excel file.")
        st.write("DEBUG: An error occurred during analysis:", e)
        st.session_state['scored_data'] = pd.DataFrame()
        st.session_state['analyze_triggered'] = False # Reset trigger on error
        st.session_state['selected_file_for_analysis'] = None # Clear selected file on error

# --- Clear Data Button ---
if st.session_state['scored_data'] is not None and not st.session_state['scored_data'].empty:
    if st.button("Clear Displayed Data", key="clear_data_button"):
        st.session_state['scored_data'] = pd.DataFrame()
        st.session_state['analyze_triggered'] = False
        st.session_state['selected_file_for_analysis'] = None
        st.success("Displayed data cleared.")
        st.rerun() # Rerun to clear the display immediately

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
    * **Machine Learning Model Hosting (SageMaker):** In a real scenario, the `get_credit_score_ml` function would call a sophisticated ML model deployed on AWS SageMaker for more accurate and robust predictions. This separates the heavy computation from the Streamlit app.
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
