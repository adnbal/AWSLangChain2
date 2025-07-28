import streamlit as st
import pandas as pd
import numpy as np
import os # To demonstrate checking environment variables if needed

# --- Dummy Credit Scoring Model ---
# In a real-world scenario, this would be a trained machine learning model (e.g., Logistic Regression, Random Forest, XGBoost)
# loaded from a file (e.g., .pkl, .joblib) or served via an API (e.g., AWS SageMaker Endpoint).
def get_credit_score(data):
    """
    A simple dummy credit scoring function.
    In a real application, this would be a sophisticated ML model.
    It takes a dictionary of input features and returns a score and a decision.
    """
    # Convert categorical inputs to numerical for this dummy model
    reason_map = {'DebtCon': 0.3, 'HomeImp': 0.2, 'Other': 0.1}
    job_map = {'Mgr': 0.2, 'Office': 0.1, 'Other': 0.05, 'ProfExe': 0.15, 'Sales': 0.1, 'Self': 0.25}

    # Assign default values for missing numerical inputs to avoid errors in dummy calculation
    loan = data.get('LOAN', 0)
    mortdue = data.get('MORTDUE', 0)
    value = data.get('VALUE', 0)
    yojs = data.get('YOJ', 0)
    derog = data.get('DEROG', 0)
    delinq = data.get('DELINQ', 0)
    clage = data.get('CLAGE', 0)
    ninq = data.get('NINQ', 0)
    clno = data.get('CLNO', 0)
    debtinc = data.get('DEBTINC', 0)
    bad = data.get('BAD', 0) # This is usually the target variable, not an input for scoring

    reason_score = reason_map.get(data.get('REASON', 'Other'), 0.1)
    job_score = job_map.get(data.get('JOB', 'Other'), 0.05)

    # Simple heuristic for a dummy score (higher is better)
    # This is highly simplified and for demonstration only.
    score = (
        (loan / 10000) * 5 + # Larger loans might imply higher trust or risk depending on context
        (value / 100000) * 10 - # Higher value of property is good
        (mortdue / 100000) * 5 + # Higher mortgage due is bad
        yojs * 0.5 - # Years on job, more is better
        derog * 20 - # Derogatory reports, more is bad
        delinq * 15 - # Delinquent lines, more is bad
        (clage / 100) * 2 + # Age of oldest credit line, more is better
        ninq * 10 - # Number of recent credit inquiries, more is bad
        clno * 1 + # Number of credit lines, more is generally good
        debtinc * 30 + # Debt-to-income ratio, higher is bad
        reason_score * 50 +
        job_score * 50
    )

    # Normalize score to a 0-100 range (example scaling)
    score = max(0, min(100, score)) # Clamp between 0 and 100

    decision = "Approved" if score >= 60 else "Rejected"
    return score, decision

# --- Streamlit UI ---
st.set_page_config(page_title="Dynamic Credit Scoring", layout="centered")

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

st.title("Dynamic Credit Scoring Application")
st.subheader("Enter applicant details to get a credit score")

# --- Accessing Secrets ---
st.sidebar.header("Application Secrets (for demonstration)")
try:
    # CORRECTED ACCESS FOR NESTED AWS KEYS
    aws_access_key = st.secrets["aws"]["access_key_id"]
    # Assuming aws_secret_access_key is also under the [aws] section
    aws_secret_key = st.secrets["aws"]["secret_access_key"]
    
    # Example for a top-level secret (if you had one)
    openai_key = st.secrets.get("openai_api_key", "NOT_SET") # Use .get() for optional secrets
    db_url = st.secrets.get("database_url", "NOT_SET")
    
    # Example for another nested section
    section_key = st.secrets.get("some_section", {}).get("key_in_section", "NOT_SET")


    st.sidebar.write(f"AWS Access Key ID: `{aws_access_key[:4]}...`") # Show only first few chars
    st.sidebar.write(f"AWS Secret Access Key: `{aws_secret_key[:4]}...`")
    st.sidebar.write(f"OpenAI API Key: `{openai_key[:4]}...`")
    st.sidebar.write(f"Database URL: `{db_url[:10]}...`")
    st.sidebar.write(f"Key in section: `{section_key}`")

    # In a real app, you would use these secrets to configure your clients:
    # import boto3
    # s3_client = boto3.client(
    #     's3',
    #     aws_access_key_id=aws_access_key,
    #     aws_secret_access_key=aws_secret_key
    # )
    # os.environ["OPENAI_API_KEY"] = openai_key # For Langchain/OpenAI
    # from langchain.llms import OpenAI
    # llm = OpenAI()

except KeyError as e:
    st.sidebar.warning(f"Secret key not found: {e}. Please ensure your `.streamlit/secrets.toml` is configured correctly or secrets are set in Streamlit Cloud.")
except Exception as e:
    st.sidebar.error(f"An error occurred while loading secrets: {e}")


# Input fields for the variables
with st.form("credit_form"):
    st.write("Please fill in the applicant's details:")

    col1, col2 = st.columns(2)
    with col1:
        loan = st.number_input("LOAN (Amount of the loan request)", min_value=0, value=10000, step=1000)
        mortdue = st.number_input("MORTDUE (Amount due on existing mortgage)", min_value=0, value=0, step=1000)
        value = st.number_input("VALUE (Value of current property)", min_value=0, value=50000, step=1000)
        reason = st.selectbox("REASON (DebtCon: Debt Consolidation, HomeImp: Home Improvement)", ["DebtCon", "HomeImp", "Other"])
        job = st.selectbox("JOB (Occupational category)", ["Mgr", "Office", "Other", "ProfExe", "Sales", "Self"])

    with col2:
        yojs = st.number_input("YOJ (Years at present job)", min_value=0, value=5, step=1)
        derog = st.number_input("DEROG (Number of major derogatory reports)", min_value=0, value=0, step=1)
        delinq = st.number_input("DELINQ (Number of delinquent credit lines)", min_value=0, value=0, step=1)
        clage = st.number_input("CLAGE (Age of oldest credit line in months)", min_value=0, value=100, step=10)
        ninq = st.number_input("NINQ (Number of recent credit inquiries)", min_value=0, value=0, step=1)
        clno = st.number_input("CLNO (Number of credit lines)", min_value=0, value=10, step=1)
        debtinc = st.number_input("DEBTINC (Debt-to-income ratio)", min_value=0.0, max_value=100.0, value=35.0, step=0.1)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Get Credit Score")

    if submitted:
        input_data = {
            'LOAN': loan,
            'MORTDUE': mortdue,
            'VALUE': value,
            'REASON': reason,
            'JOB': job,
            'YOJ': yojs,
            'DEROG': derog,
            'DELINQ': delinq,
            'CLAGE': clage,
            'NINQ': ninq,
            'CLNO': clno,
            'DEBTINC': debtinc,
            # 'BAD' is typically the target variable for training, not an input for prediction.
            # If it's historical data, it would be used for training, not direct input for scoring.
        }

        score, decision = get_credit_score(input_data)

        st.markdown(f"<div class='score-box'>Credit Score: {score:.2f}</div>", unsafe_allow_html=True)
        if decision == "Approved":
            st.markdown(f"<div class='decision-approved'>Decision: {decision}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='decision-rejected'>Decision: {decision}</div>", unsafe_allow_html=True)

st.markdown("---")
st.subheader("How this app would integrate with AWS and Langchain:")

st.markdown(
    """
    This Streamlit app provides a user interface for credit scoring. In a real-world, production-grade setup,
    the components would be distributed and managed by cloud services.

    ### ☁️ AWS Cloud Integration:
    * **Data Storage:** Input data and historical credit data would be stored in AWS S3 (for raw data) or Amazon RDS/DynamoDB (for structured data).
    * **Machine Learning Model Hosting:** The credit scoring model (e.g., a trained Scikit-learn, TensorFlow, or PyTorch model) would be deployed as an endpoint on **AWS SageMaker**. This allows the Streamlit app to make API calls to SageMaker for real-time predictions.
    * **Serverless Functions:** **AWS Lambda** could be used for pre-processing input data before sending it to SageMaker, or for post-processing the model's output.
    * **API Gateway:** If the SageMaker endpoint needs to be exposed securely or with additional logic, **AWS API Gateway** could sit in front of it.
    * **Authentication & Authorization:** **AWS Cognito** for user management and secure access to the application.
    * **Logging & Monitoring:** **AWS CloudWatch** for monitoring the application's performance and logging events.
    * **Deployment:** The Streamlit app itself could be deployed on **AWS EC2** (a virtual server), **AWS Fargate** (serverless containers), or using **Streamlit Cloud** which can connect directly to your GitHub repository.

    ### 🔗 Langchain Integration:
    Langchain is primarily used for building applications with Large Language Models (LLMs). While it's not directly for numerical credit scoring, it could enhance this application in several ways:
    * **Explainable AI (XAI):** After getting a credit score and decision, Langchain could be used to prompt an LLM to generate a human-readable explanation of *why* a particular decision was made, based on the input variables and the model's output. For example, "Your loan was rejected because your debt-to-income ratio is high and you have recent credit inquiries."
    * **Conversational Interface:** Users could interact with the app via a chat interface, asking questions about their credit score, what factors influence it, or what steps they can take to improve it. Langchain would orchestrate these conversations, potentially retrieving information from a knowledge base or calling the credit scoring model as a "tool."
    * **Automated Communication:** Langchain could help in generating personalized emails or messages to applicants based on their credit decision.

    ### 🚀 GitHub & Streamlit Cloud Deployment:
    1.  **GitHub Repository:** You would create a GitHub repository containing:
        * `app.py` (the Streamlit application code)
        * `requirements.txt` (listing `streamlit`, `pandas`, `numpy`)
        * Optionally, a `.streamlit/config.toml` for app-specific configurations (but not secrets).
    2.  **Streamlit Cloud:** You can easily deploy your Streamlit app from GitHub to [Streamlit Cloud](https://streamlit.io/cloud) by connecting your repository. Streamlit Cloud handles the hosting and environment setup. When deploying to Streamlit Cloud, you will be prompted to enter your secrets directly in their web interface, and they will be securely made available to your app via `st.secrets`.

    **To run this app locally:**
    1.  Create a folder named `.streamlit` in the same directory as `app.py`.
    2.  Inside the `.streamlit` folder, create a file named `secrets.toml` and add your secret keys as shown above.
    3.  Save the code above as `app.py`.
    4.  Ensure you have `requirements.txt` with `streamlit`, `pandas`, `numpy`.
    5.  Open your terminal, navigate to the directory where you saved the files, and run:
        ```bash
        pip install -r requirements.txt
        streamlit run app.py
        ```
    This will open the app in your web browser. You will see the secrets (partially masked) displayed in the sidebar.
    """
)
