import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("loan_prediction_model.pkl")

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Loan Prediction System"),
    html.Label("Applicant's Income"),
    dcc.Input(id="income", type="number", value=5000),
    html.Label("Loan Amount"),
    dcc.Input(id="loan_amount", type="number", value=5000),
    html.Label("Loan Term (years)"),
    dcc.Input(id="loan_term", type="number", value=5),
    html.Button("Predict", id="predict_button", n_clicks=0),
    html.Div(id="prediction_output")
])

# Define callback to update prediction result
@app.callback(
    Output("prediction_output", "children"),
    [Input("predict_button", "n_clicks")],
    [Input("income", "value"),
     Input("loan_amount", "value"),
     Input("loan_term", "value")]
)
def update_prediction(n_clicks, income, loan_amount, loan_term):
    if n_clicks > 0:
        # Preprocess input data
        data = pd.DataFrame({
            "ApplicantIncome": [income],
            "LoanAmount": [loan_amount],
            "Loan_Amount_Term": [loan_term]
        })
        # Make prediction
        prediction = model.predict(data)[0]
        return html.Div(f"Loan Status: {'Approved' if prediction == 1 else 'Rejected'}")
    else:
        return ""

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)