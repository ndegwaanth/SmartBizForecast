from flask_wtf import FlaskForm
from wtforms import StringField, EmailField, PasswordField, SubmitField, SelectField, SelectMultipleField, FloatField, IntegerField, BooleanField
from wtforms.validators import EqualTo, DataRequired, Email, Length, DataRequired, Optional, URL

class Registration(FlaskForm):
    firstname = StringField('First Name', validators=[DataRequired(), Length(min=2, max=50)])
    lastname = StringField('Last Name', validators=[DataRequired(), Length(min=2, max=50)])
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=80)])
    email = EmailField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(),
        EqualTo('password', message='Passwords must match')
    ])
    submit = SubmitField('Register')


class LoginForm(FlaskForm):
    email = EmailField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    submit = SubmitField('Log In')


# class ModelForm(FlaskForm):
#     # Dropdown for target variable
#     target = SelectField(
#         "Choose your preferred Target Variable",
#         choices=[(col, col) for col in data]
#     )

#     # Multiple selection for predictors
#     predictors = SelectMultipleField(
#         "Select your predictors",
#         choices=[(col, col) for col in data],
#         validators=[DataRequired()]
#     )

#     # Dropdown for model selection
#     model_selection = SelectField(
#         "Choose the Machine Learning Algorithm you want to use",
#         choices=[
#             ("linear_model", "LinearRegression"),
#             ("logistic_model", "LogisticRegression"),
#             ("arima_model", "ARIMA"),
#         ],
#         validators=[DataRequired()]
#     )

#     # Submit button
#     submit = SubmitField('Predict')



# Prediction forms
'''
target variable
predictors variable
hyperparameter tuning
file upload
api link
perfomance metrics
model training setting - data spliting
model preferences
'''
from wtforms import widgets  # Import widgets

class DynamicForm(FlaskForm):
    target_variable = SelectField('Target Variable', validators=[DataRequired()])

    # Use checkboxes for predictor variables
    predictor_variables = SelectMultipleField(
        'Predictor Variables',
        option_widget=widgets.CheckboxInput(),  
        widget=widgets.ListWidget(prefix_label=False), 
    )

    hyperparameter_tuning = BooleanField('Enable Hyperparameter Tuning', default=False, validators=[Optional()])
    api_link = SelectField('Action for Dataset Uploaded', choices=[
        ('customer_churn_prediction', 'Customer Churn Prediction'),
        ('sales_prediction', 'Sales Predictions')
        ])
    test_size = FloatField('Test Size (0.1 - 0.9)', default=0.2, validators=[DataRequired()])
    random_state = IntegerField('Random State', default=42, validators=[DataRequired()])
    model_preferences = SelectField('Model Preferences', choices=[
        ('linear_regression', 'Linear Regression'),
        ('logistic_regression', 'Logistic Regression'),
        ('ARIMA', 'ARIMA Model'),
    ], validators=[DataRequired()])
    submit = SubmitField('Train Model')

    def __init__(self, columns, *args, **kwargs):
        super(DynamicForm, self).__init__(*args, **kwargs)
        self.target_variable.choices = [(col, col) for col in columns]
        self.predictor_variables.choices = [(col, col) for col in columns]

