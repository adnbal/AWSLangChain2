import streamlit as st
import pandas as pd
import numpy as np
import io
import datetime
import random
import json
from collections import deque

# Mock AWS S3 client and boto3 for demonstration purposes
class MockS3Client:
    def __init__(self):
        self.buckets = {"my-langchain-demo-bucket": {}}

    def upload_fileobj(self, file_obj, bucket, key):
        if bucket not in self.buckets:
            self.buckets[bucket] = {}
        self.buckets[bucket][key] = file_obj.read()
        file_obj.seek(0) # Reset file pointer after reading
        st.success(f"File '{key}' uploaded to S3 bucket '{bucket}' successfully!")

    def list_objects_v2(self, Bucket, Prefix=''):
        if Bucket not in self.buckets:
            return {'Contents': []}
        files = []
        for key in self.buckets[Bucket]:
            if key.startswith(Prefix):
                files.append({'Key': key, 'Size': len(self.buckets[Bucket][key])})
        return {'Contents': files}

    def get_object(self, Bucket, Key):
        if Bucket not in self.buckets or Key not in self.buckets[Bucket]:
            raise Exception(f"File '{Key}' not found in bucket '{Bucket}'")
        return {'Body': io.BytesIO(self.buckets[Bucket][Key])}

# Initialize mock S3 client
s3_client = MockS3Client()

# --- Simulated ML Model (Neural Network) and Preprocessor ---
class Preprocessor:
    def __init__(self):
        # Example: scaling parameters (replace with actual trained values)
        self.mean_features = {'Age': 40, 'Income': 50000, 'LoanAmount': 15000, 'CreditScore': 700, 'EmploymentYears': 10, 'DebtToIncome': 0.3, 'NumCreditLines': 3, 'NumLatePayments': 1, 'InterestRate': 0.08, 'LoanTerm': 36, 'TLSCatPct': 0.96}
        self.std_features = {'Age': 10, 'Income': 20000, 'LoanAmount': 5000, 'CreditScore': 50, 'EmploymentYears': 5, 'DebtToIncome': 0.1, 'NumCreditLines': 1, 'NumLatePayments': 1, 'InterestRate': 0.02, 'LoanTerm': 12, 'TLSCatPct': 0.01}
        self.categorical_cols = ['Education', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose']
        self.categories = {
            'Education': ['High School', 'Bachelors', 'Masters', 'PhD'],
            'MaritalStatus': ['Single', 'Married', 'Divorced'],
            'HasMortgage': [0, 1],
            'HasDependents': [0, 1],
            'LoanPurpose': ['Auto', 'Home', 'Education', 'Debt Consolidation', 'Other']
        }

    def preprocess(self, df):
        df_processed = df.copy()

        # Handle missing values (simple imputation for demo)
        for col in self.mean_features:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(self.mean_features[col])

        # One-hot encode categorical features
        for col in self.categorical_cols:
            if col in df_processed.columns:
                for category in self.categories[col]:
                    df_processed[f"{col}_{category}"] = (df_processed[col] == category).astype(int)
                df_processed = df_processed.drop(columns=[col])

        # Scale numerical features
        for col in self.mean_features:
            if col in df_processed.columns:
                df_processed[col] = (df_processed[col] - self.mean_features[col]) / self.std_features[col]

        return df_processed

class NeuralNetworkModel:
    def __init__(self):
        # Simulate weights and biases for a simple 2-layer NN
        self.input_dim = 25 # Example: based on preprocessed features
        self.hidden_dim = 10
        self.output_dim = 1 # Binary classification (default/no default)

        # Randomly initialized weights and biases (for simulation)
        self.W1 = np.random.rand(self.input_dim, self.hidden_dim) * 0.1
        self.b1 = np.random.rand(self.hidden_dim) * 0.1
        self.W2 = np.random.rand(self.hidden_dim, self.output_dim) * 0.1
        self.b2 = np.random.rand(self.output_dim) * 0.1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500))) # Clip to prevent overflow

    def predict_proba(self, X):
        # Simple forward pass
        hidden_layer_input = np.dot(X, self.W1) + self.b1
        hidden_layer_output = np.maximum(0, hidden_layer_input) # ReLU activation

        output_layer_input = np.dot(hidden_layer_output, self.W2) + self.b2
        probabilities = self.sigmoid(output_layer_input)
        return probabilities

# Initialize preprocessor and model
preprocessor = Preprocessor()
ml_model = NeuralNetworkModel()

st.write("Simulated ML model (Neural Network) and preprocessor initialized for new data.")

# --- Data Management Section ---
st.header("Data Management")

st.write("AWS S3 client initialized!")

# Mock AWS Secrets
aws_access_key_id = "****************7UIT"
aws_secret_access_key = "************************************6THc"
s3_bucket_name = "my-langchain-demo-bucket"
aws_region = "ap-southeast-2"

st.subheader("AWS Secrets Status:")
st.write(f"AWS Access Key ID: {aws_access_key_id}")
st.write(f"AWS Secret Access Key: {aws_secret_access_key}")
st.write(f"S3 Bucket Name: {s3_bucket_name}")
st.write(f"AWS Region: {aws_region}")

st.subheader("Upload Credit Data to S3")
uploaded_file = st.file_uploader("Choose a CSV file (.csv)", type="csv", help="Drag and drop file hereLimit 200MB per file • CSV")

if uploaded_file is not None:
    # Generate a unique filename for S3
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    s3_key = f"uploads/{timestamp}_{uploaded_file.name}"

    try:
        # Simulate file upload to S3
        s3_client.upload_fileobj(uploaded_file, s3_bucket_name, s3_key)
        st.session_state['last_uploaded_s3_key'] = s3_key # Store for later analysis
    except Exception as e:
        st.error(f"Error uploading file to S3: {e}")

# --- Analyze Credit Data from S3 ---
st.header("Analyze Credit Data from S3")

# List files in the S3 bucket (mocked)
s3_files_response = s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix='uploads/')
s3_files = [obj['Key'] for obj in s3_files_response.get('Contents', [])]

if not s3_files:
    st.info("No CSV files found in S3 bucket. Please upload a file first.")
else:
    selected_file = st.selectbox("Select File for Analysis", s3_files, index=0 if 'last_uploaded_s3_key' not in st.session_state else s3_files.index(st.session_state['last_uploaded_s3_key']) if st.session_state['last_uploaded_s3_key'] in s3_files else 0)

    # Mock data for demonstration if no file is selected or uploaded
    if selected_file:
        try:
            # Simulate downloading file from S3
            obj = s3_client.get_object(Bucket=s3_bucket_name, Key=selected_file)
            df = pd.read_csv(io.BytesIO(obj['Body'].read()))
            st.success(f"File '{selected_file}' loaded successfully.")

            target_column = st.selectbox(
                "Select the Target Column (e.g., 'default', 'TARGET'):",
                options=df.columns.tolist(),
                index=df.columns.get_loc('default') if 'default' in df.columns else 0
            )

            if st.button("Analyze"):
                st.write("Analyzing data...")

                # --- Simulate Credit Scoring ---
                def get_credit_score_ml(data_df):
                    # Simulate preprocessing
                    processed_data = preprocessor.preprocess(data_df)

                    # Ensure all expected features are present, fill with 0 if not
                    # This is a simplification; in a real model, you'd align columns
                    expected_features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'EmploymentYears',
                                         'DebtToIncome', 'NumCreditLines', 'NumLatePayments', 'InterestRate',
                                         'LoanTerm', 'TLSCatPct', 'Education_High School', 'Education_Bachelors',
                                         'Education_Masters', 'Education_PhD', 'MaritalStatus_Single',
                                         'MaritalStatus_Married', 'MaritalStatus_Divorced', 'HasMortgage_0',
                                         'HasMortgage_1', 'HasDependents_0', 'HasDependents_1', 'LoanPurpose_Auto',
                                         'LoanPurpose_Home', 'LoanPurpose_Education', 'LoanPurpose_Debt Consolidation',
                                         'LoanPurpose_Other']

                    # Ensure processed_data has all expected columns, filling missing with 0
                    for feature in expected_features:
                        if feature not in processed_data.columns:
                            processed_data[feature] = 0
                    processed_data = processed_data[expected_features] # Reorder to match model's expected input

                    # Simulate prediction
                    probabilities = ml_model.predict_proba(processed_data.values)
                    predictions = (probabilities > 0.5).astype(int).flatten() # Binary prediction

                    # Simulate risk categories
                    risk_scores = np.random.uniform(0.1, 0.9, len(data_df)) # Random risk score
                    risk_categories = np.where(predictions == 1, 'High Risk', 'Low Risk')

                    return pd.DataFrame({
                        'prediction': predictions,
                        'risk_score': risk_scores,
                        'risk_category': risk_categories
                    })

                # Perform analysis
                if target_column in df.columns:
                    features_df = df.drop(columns=[target_column])
                else:
                    features_df = df.copy() # If target column not found, use all columns

                scored_data = get_credit_score_ml(features_df)
                st.session_state['scored_data'] = scored_data
                st.session_state['original_data'] = df # Store original data for dashboard

                st.write("Analysis complete. Results:")
                st.dataframe(scored_data.head())

                # --- Explainable AI (XAI) with LLM (Simulated) ---
                st.subheader("Explainable AI (XAI) for High-Risk Applicants")
                high_risk_applicants = st.session_state['original_data'][st.session_state['scored_data']['risk_category'] == 'High Risk']

                if not high_risk_applicants.empty:
                    st.write("Generating explanations for high-risk applicants...")
                    explanations = []
                    for index, row in high_risk_applicants.head(5).iterrows(): # Explain top 5 for demo
                        # Simulate LLM call for explanation
                        # In a real scenario, you'd send relevant features to an LLM
                        # and get a human-readable explanation.
                        simulated_explanation = (
                            f"Applicant {index+1} was flagged as high-risk due to factors such as "
                            f"a relatively low Credit Score ({row['CreditScore']}), "
                            f"a high Debt-to-Income ratio ({row['DebtToIncome']:.2f}), "
                            f"and a higher number of late payments ({row['NumLatePayments']}). "
                            f"Their Loan Amount of ${row['LoanAmount']:,} also contributed to the risk assessment."
                        )
                        explanations.append(f"**Applicant {index+1}:** {simulated_explanation}")
                    for exp in explanations:
                        st.markdown(exp)
                else:
                    st.info("No high-risk applicants identified in the analyzed data.")

        except Exception as e:
            st.error(f"Error loading or processing file from S3: {e}")
    else:
        st.info("Please select a file to analyze.")

# --- Credit Scoring Dashboard ---
st.header("Credit Scoring Dashboard")

if 'scored_data' in st.session_state and not st.session_state['scored_data'].empty:
    st.write("Displaying Credit Scoring Dashboard:")

    scored_data = st.session_state['scored_data']
    original_data = st.session_state['original_data']

    # Combine for dashboard display
    dashboard_df = pd.concat([original_data.reset_index(drop=True), scored_data.reset_index(drop=True)], axis=1)

    st.subheader("Overall Risk Distribution")
    risk_counts = dashboard_df['risk_category'].value_counts()
    st.bar_chart(risk_counts)

    st.subheader("Risk Score Distribution")
    st.hist_chart(dashboard_df['risk_score'])

    st.subheader("Detailed Scored Data")
    st.dataframe(dashboard_df)

    # Example: Filter by risk category
    selected_risk_category = st.selectbox("Filter by Risk Category:", ['All', 'High Risk', 'Low Risk'])
    if selected_risk_category == 'All':
        filtered_df = dashboard_df
    else:
        filtered_df = dashboard_df[dashboard_df['risk_category'] == selected_risk_category]
    st.dataframe(filtered_df)

else:
    st.write("Upload a CSV file and click 'Analyze' to view the dashboard.")
    st.write("DEBUG: scored_data is empty, so dashboard is not displayed.")


st.header("How this app would integrate with AWS and LLM (Recap):")
st.markdown("""
This Streamlit app demonstrates uploading and analyzing sophisticated credit data from AWS S3, leveraging a powerful Neural Network machine learning model.

☁️ **AWS Cloud Integration:**
* **Data Storage (S3):** Data files are stored in AWS S3, providing durable and scalable storage.
* **Machine Learning Model Hosting (SageMaker):** In a real scenario, the `get_credit_score_ml` function would call a sophisticated ML model deployed on AWS SageMaker for more accurate and robust predictions. This separates the heavy computation from the Streamlit app.
* **Serverless Functions (Lambda):** Could be used for automated processing of new files uploaded to S3 (e.g., triggering the scoring process automatically).
* **Authentication & Authorization (Cognito):** For secure user access to the app and S3, ensuring only authorized users can upload or view sensitive data.
* **Logging & Monitoring (CloudWatch):** To track app performance, S3 interactions, and potential errors, providing insights for operational management.

🧠 **Large Language Model (LLM) Integration:**

The LLM (currently OpenAI's GPT-3.5-turbo) is used to provide Explainable AI (XAI). After identifying rejected applicants, the LLM generates concise, human-readable explanations for why specific applicants were flagged as high-risk, based on their input features. This helps in understanding and communicating complex model decisions.
""")
