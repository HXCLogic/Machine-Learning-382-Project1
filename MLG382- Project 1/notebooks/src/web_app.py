def app():
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    import joblib
    import pandas as pd

    # Load the trained model
    model = joblib.load("../artifacts/model_1.pkl")

    # Initialize the Dash app
    app = dash.Dash(__name__)

    # Define the layout of the app
    app.layout = html.Div([
        html.H1("Loan Prediction System"),
        html.Label("Gender"),
        dcc.Dropdown(
            id="gender-dropdown",
            options=[
                {'label': 'Male', 'value': 'Male'},
                {'label': 'Female', 'value': 'Female'}
            ],
            value='Male'
        ),
        html.Label("Married"),
        dcc.Dropdown(
            id="married-dropdown",
            options=[
                {'label': 'Yes', 'value': 'Yes'},
                {'label': 'No', 'value': 'No'}
            ],
            value='No'
        ),
        html.Label("Dependents"),
        dcc.Input(id="dependents-number", 
            type="number", 
            value=0
        ),
        html.Label("Education"),
        dcc.Dropdown(
            id="education-dropdown",
            options=[
                {'label': 'Graduate', 'value': 'Graduate'},
                {'label': 'Not Graduate', 'value': 'Not Graduate'}
            ],
            value='Not Graduate'
        ),
        html.Label("Self Employed"),
        dcc.Dropdown(
            id="self-employed-dropdown",
            options=[
                {'label': 'Yes', 'value': 'Yes'},
                {'label': 'No', 'value': 'No'}
            ],
            value='No'
        ),
        html.Label("Applicant's Income"),
        dcc.Input(id="applicantIncome", type="number", value=5000),
        html.Label("Co-Applicant's Income"),
        dcc.Input(id="coapplicantIncome", type="number", value=0),
        html.Label("Loan Amount"),
        dcc.Input(id="loan_amount", type="number", value=120),
        html.Label("Loan Term (months)"),
        dcc.Input(id="loan_term", type="number", value=360),
        html.Label("Credit History"),
        dcc.Dropdown(
            id="credit_history-dropdown",
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ),
        html.Label("Property Area"),
        dcc.Dropdown(
            id="property_area-dropdown",
            options=[
                {'label': 'Urban', 'value': 'Urban'},
                {'label': 'Semiurban', 'value': 'Semiurban'},
                {'label': 'Rural', 'value': 'Rural'}
            ],
            value='Urban'
        ),
        html.Button("Predict", id="predict_button", n_clicks=0),
        html.Div(id="prediction_output")
    ])

    # Define callback to update prediction result
    @app.callback(
        Output("prediction_output", "children"),
        [Input("predict_button", "n_clicks")],
        [Input("gender-dropdown", "value"),
        Input("married-dropdown", "value"),
        Input("dependents-number", "value"),
        Input("education-dropdown", "value"),
        Input("self-employed-dropdown", "value"),
        Input("applicantIncome", "value"),
        Input("coapplicantIncome", "value"),
        Input("loan_amount", "value"),
        Input("loan_term", "value"),
        Input("credit_history-dropdown", "value"),
        Input("property_area-dropdown", "value")]
    )
    def update_prediction(n_clicks, gender, married, dependents, education, self_employed, income, co_income, loan_amount, loan_term, credit_history, property_area):
        if n_clicks > 0:
            # Preprocess input data
            data = pd.DataFrame({
                "Gender": [gender],
                "Married": [married],
                "Dependents": [dependents],
                "Education": [education],
                "Self_Employed": [self_employed],
                "ApplicantIncome": [income],
                "CoapplicantIncome": [co_income],
                "LoanAmount": [loan_amount],
                "Loan_Amount_Term": [loan_term],
                "Credit_History": [credit_history],
                "Property_Area": [property_area]
            })
            # Make prediction
            prediction = model.predict(data)[0]
            return html.Div(f"Loan Status: {'Approved' if prediction == 1 else 'Rejected'}")
        else:
            return ""

    # Run the app
    if __name__ == "__main__":
        app.run_server(debug=True)

    return app