def prepare(path: str):
    # Importing the libraries
    import numpy as np
    import pandas as pd

    # Importing the dataset
    df = pd.read_csv(f'../data/{path}')

    # Remove irrelevant columns
    df.drop(columns='Loan_ID', inplace=True)

    return df