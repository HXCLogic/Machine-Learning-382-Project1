import numpy as np
import pandas as pd

def model_1(df: pd.DataFrame):
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

    # Model and Predict
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

def model_2(df: pd.DataFrame):
    # Feature Engineering
    df['Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df.drop(columns=['ApplicantIncome', 'CoapplicantIncome'], inplace=True)

    # Scale the numerical features
    cols = ['Income', 'LoanAmount', 'Loan_Amount_Term']

    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    df[cols] = ss.fit_transform(df[cols])

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=['Loan_Status'], inplace=False)
    y = df['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Model and Predict
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