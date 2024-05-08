import numpy as np
import pandas as pd

def model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame):
    from sklearn.metrics import accuracy_score

    # Create an instance of the DecisionTreeClassifier
    from sklearn.tree import DecisionTreeClassifier
    model_dt = DecisionTreeClassifier()

    # Fit the model to the training data
    model_dt.fit(X_train, y_train)

    # Evaluate the model on the testing data
    y_pred = model_dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Decision Tree Accuracy:", (accuracy*100).__round__(4))


    # Create an instance of the Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    model_rf = RandomForestClassifier()

    # Fit the model to the training data
    model_rf.fit(X_train, y_train)

    # Evaluate the model on the testing data
    y_pred = model_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Random Forest Accuracy:", (accuracy*100).__round__(4))

    # Create an instance of the Logistic Regression
    from sklearn.linear_model import LogisticRegression
    model_lr = LogisticRegression()

    # Fit the model to the training data
    model_lr.fit(X_train, y_train)

    # Evaluate the model on the testing data
    y_pred = model_lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Logistic Regression Accuracy:", (accuracy*100).__round__(4))

    return model_dt, model_rf, model_lr