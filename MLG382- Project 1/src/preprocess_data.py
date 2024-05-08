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

    # One Hot Encoding Categorical Columns
    df['Dependents'] = df['Dependents'].replace(to_replace='3+', value='4')

    df['Gender'] = df['Gender'].map({'Male' : 1, 'Female' : 0}).astype('int')
    df['Married'] = df['Married'].map({'Yes' : 1, 'No' : 0}).astype('int')
    df['Education'] = df['Education'].map({'Graduate' : 1, 'Not Graduate' : 0}).astype('int')
    df['Self_Employed'] = df['Self_Employed'].map({'Yes' : 1, 'No' : 0}).astype('int')
    df['Property_Area'] = df['Property_Area'].map({'Rural' : 0, 'Urban' : 1, 'Semiurban': 2}).astype('int')
    df['Loan_Status'] = df['Loan_Status'].map({'Y' : 1, 'N' : 0}).astype('int')

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

    # Scale the numerical features
    cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    df[cols] = ss.fit_transform(df[cols])

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=['Loan_Status'], inplace=False)
    y = df['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    return X_train, X_test, y_train, y_test