import numpy as np
import pandas as pd

def preprocess(df: pd.DataFrame):
    # Remove rows with missing values
    df.dropna(subset=['Gender', 'Dependents', 'LoanAmount', 'Loan_Amount_Term'], inplace=True)

    # Handle duplicate rows
    df.drop_duplicates()

    # Fill categorical values
    df['Self_Employed']=df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

    # Fill numerical values
    df['Credit_History'].fillna(df['Credit_History'].mean(), inplace=True)

    # Create a mask to filter out the outliers for 'ApplicantIncome'
    mask_ApplicantIncome = df['ApplicantIncome'] <= 7441

    # Create a mask to filter out the outliers for 'CoapplicantIncome'
    mask_CoapplicantIncome = df['CoapplicantIncome'] <= 5302

    # Create a mask to filter out the outliers for 'LoanAmount'
    mask_LoanAmount1 = df['LoanAmount'] >= 25
    mask_LoanAmount2 = df['LoanAmount'] <= 230

    # Create a mask to filter out the outliers for 'LoanAmount'
    mask_Loan_Amount_Term = df['Loan_Amount_Term'] == 360

    # Filter out the outliers from the dataframe
    df = df[mask_ApplicantIncome & mask_CoapplicantIncome & mask_LoanAmount1 & mask_LoanAmount2 & mask_Loan_Amount_Term]

    # Scaling
    df['Loan_Status'] = df['Loan_Status'].map({'Y' : 1, 'N' : 0}).astype('int')

    return df