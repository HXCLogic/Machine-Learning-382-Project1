import numpy as np
import pandas as pd

def model(df: pd.DataFrame):
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=['ChurnLikelihood'], inplace=False)
    y = df['ChurnLikelihood']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Model and Predict
    from sklearn.pipeline import make_pipeline
    from category_encoders import OneHotEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    
    model = make_pipeline(
        OneHotEncoder(use_cat_names=True), # encode cat features
        StandardScaler(), # imputation
        LogisticRegression()) # build model

    # fit the model
    model.fit(X_train, y_train)

    # Accuracy of model
    from sklearn.metrics import accuracy_score
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Logistic Regression:", (accuracy*100).__round__(4))

    from sklearn.metrics import mean_absolute_error
    # Predict the train data
    y_pred_training = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Compute MAE
    print("Training MAE:", round(mean_absolute_error(y_train, y_pred_training),2))
    print("Test data MAE:", round(mean_absolute_error(y_test, y_pred_test),2))

    return model