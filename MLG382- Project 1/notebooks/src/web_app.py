def app():
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    import joblib
    import pandas as pd

    # Load the trained model
    model = joblib.load("../artifacts/model.pkl")

    # Initialize the Dash app
    app = dash.Dash(__name__, external_stylesheets=['../../style.css'])

    # Define the layout of the app
    app.layout = html.Div([
        html.H1("Employee Churn Prediction", style={'color': 'blue', 'font-size': '24px'}),

        html.Label("Branch"),
        dcc.Dropdown(
            id="branch",
            options=[
                {'label': 'San Francisco', 'value': 'San Francisco'},
                {'label': 'Chicago', 'value': 'Chicago'},
                {'label': 'Miami', 'value': 'Miami'},
                {'label': 'Scranton', 'value': 'Scranton'},
                {'label': 'Boston', 'value': 'Boston'},
                {'label': 'New York', 'value': 'New York'},
                {'label': 'Philadelphia', 'value': 'Philadelphia'},
                {'label': 'Los Angeles', 'value': 'Los Angeles'},
                {'label': 'Seattle', 'value': 'Seattle'},
                {'label': 'Atlanta', 'value': 'Atlanta'},
                {'label': 'Denver', 'value': 'Denver'},
                {'label': 'Dallas', 'value': 'Dallas'}
            ],
            value='San Francisco'
        ),

        html.Label("Tenure"),
        dcc.Input(id="tenure", type="number", value=4.0),

        html.Label("Salary"),
        dcc.Input(id="salary", type="number", value=63000.0),

        html.Label("Department"),
        dcc.Dropdown(
            id="department",
            options=[
                {'label': 'Legal', 'value': 'Legal'},
                {'label': 'Accounting', 'value': 'Accounting'},
                {'label': 'Quality Assurance', 'value': 'Quality Assurance'},
                {'label': 'Customer Service', 'value': 'Customer Service'},
                {'label': 'Sales', 'value': 'Sales'},
                {'label': 'Administration', 'value': 'Administration'},
                {'label': 'Facilities Management', 'value': 'Facilities Management'},
                {'label': 'Research and Development', 'value': 'Research and Development'},
                {'label': 'Operations', 'value': 'Operations'},
                {'label': 'Marketing', 'value': 'Marketing'},
                {'label': 'Public Relations', 'value': 'Public Relations'},
                {'label': 'IT Support', 'value': 'IT Support'},
                {'label': 'Procurement', 'value': 'Procurement'},
                {'label': 'Product Management', 'value': 'Product Management'},
                {'label': 'Human Resources', 'value': 'Human Resources'}
            ],
            value='Legal'
        ),

        html.Label("JobSatisfaction"),
        dcc.Input(id="job", type="number", value=3.0),

        html.Label("WorkLifeBalance"),
        dcc.Input(id="balance", type="number", value=3.0),

        html.Label("CommuteDistance"),
        dcc.Dropdown(
            id="commute",
            options=[
                {'label': 'Short', 'value': 'Short'},
                {'label': 'Medium', 'value': 'Medium'},
                {'label': 'Long', 'value': 'Long'}
            ],
            value='Long'
        ),

        html.Label("MaritalStatus"),
        dcc.Dropdown(
            id="married",
            options=[
                {'label': 'Single', 'value': 'Single'},
                {'label': 'Married', 'value': 'Married'},
            ],
            value='Married'
        ),

        html.Label("Education"),
        dcc.Dropdown(
            id="education",
            options=[
                {'label': 'High School', 'value': 'High School'},
                {'label': 'Bachelor', 'value': 'Bachelor'},
                {'label': 'Master', 'value': 'Master'},
                {'label': 'Doctor', 'value': 'Doctor'}
            ],
            value='High School'
        ),

        html.Label("PerformanceRating"),
        dcc.Input(id="rating", type="number", value=3.0),

        html.Label("TrainingHours"),
        dcc.Input(id="training", type="number", value=88.0),

        html.Label("NumProjects"),
        dcc.Input(id="projects", type="number", value=3.0),

        html.Label("YearsSincePromotion"),
        dcc.Input(id="promotion", type="number", value=0.0),

        html.Label("EnvironmentSatisfaction"),
        dcc.Input(id="satisfaction", type="number", value=2.0),

        html.Button("Predict", id="predict_button", n_clicks=0),
        html.Div(id="prediction_output", style={'color': 'red'})
    ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '1em'})

    # Define callback to update prediction result
    @app.callback(
        Output("prediction_output", "children"),
        [Input("predict_button", "n_clicks")],
        
        [Input("branch", "value"),
            Input("tenure", "value"),
            Input("salary", "value"),
            Input("department", "value"),
            Input("job", "value"),
            Input("balance", "value"),
            Input("commute", "value"),
            Input("married", "value"),
            Input("education", "value"),
            Input("rating", "value"),
            Input("training", "value"),
            Input("projects", "value"),
            Input("promotion", "value"),
            Input("satisfaction", "value")]
    )
    def update_prediction(n_clicks, branch, tenure, salary, department, job, balance, commute, married, education, rating, training, projects, promotion, satisfaction):
        if n_clicks > 0:
            # Preprocess input data
            data = pd.DataFrame({
                "Branch": [branch],
                "Tenure": [tenure],
                "Salary": [salary],
                "Department": [department],
                "JobSatisfaction": [job],
                "WorkLifeBalance": [balance],
                "CommuteDistance": [commute],
                "MaritalStatus": [married],
                "Education": [education],
                "PerformanceRating": [rating],
                "TrainingHours": [training],
                "NumProjects": [projects],
                "YearsSincePromotion": [promotion],
                "EnvironmentSatisfaction": [satisfaction]
            })
            # Make prediction
            prediction = model.predict(data)[0]
            
            if prediction == 2:
                prediction_str = 'Highly Likely to Churn'
            elif prediction == 1:
                prediction_str = 'Moderately Likely to Churn'
            else:
                prediction_str = 'Slightly Likely to Churn'

            return html.Div(f"Churn Likelihood: {prediction_str}")
        else:
            return ""

    # Run the app
    if __name__ == "__main__":
        app.run_server(debug=True)

    return app