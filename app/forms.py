from flask_wtf import FlaskForm
from wtforms import StringField, EmailField, PasswordField, SubmitField, SelectField, SelectMultipleField, FloatField, IntegerField, BooleanField
from wtforms.validators import EqualTo, DataRequired, Email, Length, DataRequired, Optional, URL
from wtforms import ValidationError, validators

class Registration(FlaskForm):
    firstname = StringField('First Name', validators=[DataRequired(), Length(min=2, max=50)])
    lastname = StringField('Last Name', validators=[DataRequired(), Length(min=2, max=50)])
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=25)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=8, message='Password must be at least 8 characters')
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(),
        EqualTo('password', message='Passwords must match')
    ])
    submit = SubmitField('Sign Up')


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
    
    predictor_variables = SelectMultipleField(
        'Predictor Variables',
        validators=[DataRequired(message="Please select at least one predictor")],
        choices=[],
        render_kw={'class': 'form-select', 'size': '8'}
    )
    
    hyperparameter_tuning = BooleanField('Enable Hyperparameter Tuning', default=False)
    test_size = FloatField('Test Size', default=0.2, validators=[
        DataRequired(),
        validators.NumberRange(min=0.1, max=0.9, message="Must be between 0.1 and 0.9")
    ])
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
        # Remove api_link if not needed
        if hasattr(self, 'api_link'):
            del self.api_link

    def validate(self, extra_validators=None):
        # First run default validation
        initial_validation = super(DynamicForm, self).validate()
        if not initial_validation:
            return False
            
        # Custom validation
        if self.target_variable.data in self.predictor_variables.data:
            self.target_variable.errors.append("Target variable cannot be a predictor")
            return False
            
        return True

class ForgotPasswordForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Generate Reset Link')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('New Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password', message='Passwords must match.')])
    submit = SubmitField('Reset Password')