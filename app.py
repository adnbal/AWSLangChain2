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
        'TLSatPct': [0.95, 0.98, 0.96, 0.97, 0.94, 0.97, 0.95
