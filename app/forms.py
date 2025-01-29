from flask_wtf import FlaskForm
from wtforms import StringField, EmailField, PasswordField, SubmitField, SelectField, SelectMultipleField
from wtforms.validators import EqualTo, DataRequired, Email, Length
from routes import Data

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


class ModelForm(FlaskForm):
    # Dropdown for target variable
    target = SelectField(
        "Choose your preferred Target Variable",
        choices=[(col, col) for col in datacolumns]  # Each choice must be a tuple # type: ignore
    )

    # Multiple selection for predictors
    predictors = SelectMultipleField(
        "Select your predictors",
        choices=[(col, col) for col in columns],  # Each choice must be a tuple # type: ignore
        validators=[DataRequired()]
    )

    # Dropdown for model selection
    model_selection = SelectField(
        "Choose the Machine Learning Algorithm you want to use",
        choices=[
            ("linear_model", "LinearRegression"),
            ("logistic_model", "LogisticRegression"),
            ("arima_model", "ARIMA"),
        ],
        validators=[DataRequired()]
    )

    # Submit button
    submit = SubmitField('Predict')