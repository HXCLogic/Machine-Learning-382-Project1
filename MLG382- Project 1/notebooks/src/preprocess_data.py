import numpy as np
import pandas as pd

def preprocess(df: pd.DataFrame):
    # Remove rows with missing values for Churned
    df.dropna(subset='ChurnLikelihood', inplace=True)

    # Handle duplicate rows
    df.drop_duplicates()

    # Handling high and low cardinality features
    high_cardinality = ['OverTime']
    df.drop(columns=high_cardinality, inplace=True)

    # Fill categorical values
    for col in df.select_dtypes(include=['object']).columns:
        df[col]=df[col].fillna(df[col].mode()[0])

    # Fill numerical values
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col].fillna(df[col].mean(), inplace=True)

    # Create a mask to filter out the outliers for 'Tenure'
    mask_Tenure = df['Tenure'] <= 17

    # Create a mask to filter out the outliers for 'Salary'
    mask_Salary1 = df['Salary'] >= 44_500
    mask_Salary2 = df['Salary'] <= 88_000

    # Create a mask to filter out the outliers for 'JobSatisfaction'
    mask_JobSatisfaction = df['JobSatisfaction'] >= 2

    # Create a mask to filter out the outliers for 'WorkLifeBalance'
    mask_WorkLifeBalance1 = df['WorkLifeBalance'] >= 2.840207
    mask_WorkLifeBalance2 = df['WorkLifeBalance'] <= 4.712464

    # Create a mask to filter out the outliers for 'PerformanceRating'
    mask_PerformanceRating1 = df['PerformanceRating'] >= 2.835084
    mask_PerformanceRating2 = df['PerformanceRating'] <= 4.13773

    # Create a mask to filter out the outliers for 'TrainingHours'
    mask_TrainingHours = df['TrainingHours'] <= 57

    # Create a mask to filter out the outliers for 'NumProjects'
    mask_NumProjects = df['NumProjects'] <= 5.4

    # Create a mask to filter out the outliers for 'YearsSincePromotion'
    mask_YearsSincePromotion = df['YearsSincePromotion'] <= 2

    # Create a mask to filter out the outliers for 'EnvironmentSatisfaction'
    mask_EnvironmentSatisfaction = df['EnvironmentSatisfaction'] <= 4.2

    # Filter out the outliers from the dataframe
    df = df[mask_Tenure & mask_Salary1 & mask_Salary2 & mask_JobSatisfaction & mask_WorkLifeBalance1 & mask_WorkLifeBalance2 & mask_PerformanceRating1 & mask_PerformanceRating2 & mask_TrainingHours & mask_NumProjects & mask_YearsSincePromotion & mask_EnvironmentSatisfaction]

    # Scaling
    df['ChurnLikelihood'] = df['ChurnLikelihood'].map({'Slightly Likely to Churn' : 0, 'Moderately Likely to Churn' : 1, 'Highly Likely to Churn' : 2}).astype('int')

    return df