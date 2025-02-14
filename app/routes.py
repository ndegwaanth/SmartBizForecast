from .forms import Registration, LoginForm, DynamicForm
from flask_bcrypt import Bcrypt
from flask_mysqldb import MySQL
import secrets
from flask import session, render_template, redirect, request, Blueprint, url_for, flash
import pandas as pd
from werkzeug.utils import secure_filename
import os
from .models import User
from .cleaning import data_cleaning
from dotenv import load_dotenv
# from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from flask_login import login_user, logout_user, login_required, login_remembered, current_user
import os
import subprocess
import uuid

load_dotenv()


key = secrets.token_hex(64)

main_bp = Blueprint('main', __name__)
bcrypt = Bcrypt()



@main_bp.route('/')
def landing_page():
    return render_template('landing_page.html')

@main_bp.route('/homepage')
def homepage():
    login_remembered()
    return render_template('homepage.html')

@main_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.login'))


@main_bp.route("/signup", methods=["POST", "GET"])
def register():
    from . import collection
    form = Registration()
    if form.validate_on_submit() and request.method == 'POST':
        # Collect form data
        firstname = form.firstname.data
        lastname = form.lastname.data
        username_reg = form.username.data
        email_reg = form.email.data
        password_reg = form.password.data
        confirm_password = form.confirm_password.data

        # Check if passwords match
        if password_reg != confirm_password:
            return render_template("register.html", form=form, error="Passwords do not match")

        # Hash the password
        password_hash = bcrypt.generate_password_hash(password_reg).decode('utf-8')

        try:
            # Check if the email or username already exists
            if collection.find_one({"Email": email_reg}) or collection.find_one({"Username": username_reg}):
                return render_template("register.html", form=form, error="Email or username already taken")

            # Insert the new user into MongoDB
            user_data = {
                "FirstName": firstname,
                "LastName": lastname,
                "Username": username_reg,
                "Email": email_reg,
                "Password": password_hash
            }
            collection.insert_one(user_data)

            print("Connection Successful")

            # Automatically log in the user after registration
            user_dict = collection.find_one({"Email": email_reg})
            user = User(user_dict)
            login_user(user)

            flash(f"Welcome, {current_user.username}!", "success")
            return redirect(url_for("main.homepage"))
        except Exception as e:
            return render_template("register.html", form=form, error="Database error: " + str(e))
    print("Connection failled")
    return render_template("register.html", form=form)


@main_bp.route('/login', methods=['POST', 'GET'])
def login():
    from . import collection
    form = LoginForm()
    if form.validate_on_submit():
        # Collecting form data
        email = form.email.data
        password = form.password.data

        try:
            # Finding the user in MongoDB by email
            user_dict = collection.find_one({"Email": email})
            
            if user_dict and bcrypt.check_password_hash(user_dict["Password"], password):
                user = User(user_dict)
                login_user(user)
                flash(f"Welcome back, {current_user.username}!", "success")
                return redirect(url_for('main.homepage'))
            else:
                return render_template("login.html", form=form, error="Invalid email or password")
        except Exception as e:
            return render_template("login.html", form=form, error="Database error: " + str(e))
    return render_template("login.html", form=form)


# Directory where I will store the uploaded files
UPLOAD_FOLDER = 'data/user'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

# Ensuring the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@main_bp.route('/upload', methods=['POST', 'GET'])
def upload_data():
    if request.method == 'POST':
        file = request.files.get('file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(file_path)

            try:
                if filename.endswith('.csv'):
                    data = pd.read_csv(file_path, nrows=5)
                    print(data.head(10))
                elif filename.endswith('.xlsx'):
                    data = pd.read_excel(file_path, nrows=5)
                    print(data.head(10))
                else:
                    flash("Unsupported file format.", "error")
                    return redirect(url_for('main.upload_data'))

                # Prepare data for the template
                table_headers = data.columns.tolist()
                table_data = data.values.tolist()

                # Create the dynamic form
                session['columns'] = table_headers
                session['uploaded_file'] = unique_filename
                # session['data_rows'] = data.values.tolist()
                form = DynamicForm(columns=table_headers)
                

                # # Pass data and form to prediction.html
                # return render_template(
                #     'homepage.html',
                #     headers=table_headers,
                #     rows=table_data,
                #     form=form
                # )

                # Passing the data to the template
                return render_template('prediction.html', headers=table_headers, rows=table_data, form=form)

            except Exception as e:
                flash(f"Error processing file: {str(e)}", "error")
                return redirect(url_for('main.upload_data'))
        else:
            flash("Invalid file format. Please upload a .csv or .xlsx file.", "error")
            return redirect(url_for('main.upload_data'))

    return render_template('homepage.html')


@main_bp.route('/prediction')
def prediction():
    columns = session.get('columns', [])
    form = DynamicForm(columns=columns)

    filename = session.get('uploaded_file')
    rows = []

    if filename:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if filename.endswith('.csv'):
                data = pd.read_csv(file_path, nrows=10)
            elif filename.endswith('.xlsx'):
                data = pd.read_excel(file_path, nrows=10)

            rows = data.values.tolist()
        except Exception as e:
            flash(f"Error rading uploaded file: {str(e)}", "error")
    
    return render_template('prediction.html', form=form, headers=columns, rows=rows)

@main_bp.route('/train_model', methods=['POST'])
def train_model():
    columns = session.get('columns', [])
    form = DynamicForm(columns=columns)

    if form.validate_on_submit():
        # Extracting form data
        target_variable = form.target_variable.data
        predictor_variables = form.predictor_variables.data
        hyperparameter_tuning = form.hyperparameter_tuning.data
        api_link = form.api_link.data
        performance_metrics = form.performance_metrics.data
        test_size = form.test_size.data
        random_state = form.random_state.data
        model_preferences = form.model_preferences.data

        # Performing model training here
        flash("Model training initiated!", "success")
        return redirect(url_for('main.prediction'))

    flash("Form validation failed. Please check your inputs.", "error")
    return redirect(url_for('main.prediction'))


def run_streamlit():
    streamlit_script = os.path.join(os.path.dirname(__file__), "dashboard.py")
    subprocess.Popen(
        ["streamlit", "run", streamlit_script, "--server.port", "8501", "--server.address", "0.0.0.0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

@main_bp.route("/streamlit")
def streamlit_redirect():
    return redirect("http://localhost:8501", code=302)