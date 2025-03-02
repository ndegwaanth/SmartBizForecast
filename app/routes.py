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
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from .predictions import ChurnPrediction
from flask import jsonify

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

        # Check if a file is selected
        if not file:
            flash("No file was uploaded.", "error")
            print("DEBUG: No file received in request")
            return redirect(url_for('main.upload_data'))

        # Check if the file is empty or has an invalid format
        if file.filename == '':
            flash("No selected file.", "error")
            print("DEBUG: Empty filename")
            return redirect(url_for('main.upload_data'))

        if not allowed_file(file.filename):
            flash("Invalid file format. Please upload a .csv or .xlsx file.", "error")
            print(f"DEBUG: Invalid file type -> {file.filename}")
            return redirect(url_for('main.upload_data'))

        # Save file with a unique name
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)

        try:
            # Read the dataset (Only first 10 rows to avoid session overload)
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(file_path, nrows=10)
            elif filename.lower().endswith('.xlsx'):
                df = pd.read_excel(file_path, nrows=10)
            else:
                flash("Unsupported file format.", "error")
                print("DEBUG: File format not supported")
                return redirect(url_for('main.upload_data'))

            print("DEBUG: File uploaded successfully:", file_path)
            print("DEBUG: Data preview:\n", df.head())

            # Store only column names in session, not the full dataset
            session['columns'] = df.columns.tolist()
            session['uploaded_file'] = unique_filename
            session['dataset_preview'] = df.head(5).to_json()  # Store a preview, not full data

            # Generate dynamic form
            form = DynamicForm(columns=df.columns.tolist())

            return render_template('prediction.html', headers=df.columns.tolist(), rows=df.values.tolist(), form=form)

        except Exception as e:
            flash(f"Error processing file: {str(e)}", "error")
            print("DEBUG: Exception ->", str(e))
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


def run_streamlit():
    streamlit_script = os.path.join(os.path.dirname(__file__), "dashboard.py")
    subprocess.Popen(
        ["streamlit", "run", streamlit_script, "--server.port", "8501", "--server.address", "0.0.0.0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

@main_bp.route("/streamlit")
def streamlit_redirect():
    return redirect("http://localhost:8501", code=302)


# Customer Churn Prediction
# Initialize churn prediction model
churn = ChurnPrediction()

@main_bp.route('/predict', methods=['GET', 'POST'])
def model_training():
    columns = session.get('columns', [])  # Retrieve column names from session
    form = DynamicForm(columns=columns)
    metrics = None 
    graphs = []  # Initialize empty graph list

    if form.validate_on_submit():
        target_variable = form.target_variable.data
        predictor_variables = form.predictor_variables.data
        test_size = float(form.test_size.data or 0.2)  # Default 20% test size
        random_state = int(form.random_state.data or 42)  # Default seed

        # Ensure dataset exists
        uploaded_file = session.get('uploaded_file')
        if not uploaded_file:
            flash("No dataset found. Please upload a dataset first.", "warning")
            return redirect(url_for('main.upload_data'))

        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file)

        # Reload dataset from disk
        try:
            if uploaded_file.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif uploaded_file.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                flash("Unsupported file format.", "error")
                return redirect(url_for('main.upload_data'))
        except Exception as e:
            flash(f"Error loading dataset: {str(e)}", "error")
            return redirect(url_for('main.upload_data'))

        # Validate selected columns
        missing_columns = [col for col in predictor_variables if col not in df.columns]
        if target_variable not in df.columns or missing_columns:
            flash(f"Invalid columns: {missing_columns} missing.", "danger")
            return redirect(url_for('main.upload_data'))

        # Train the model and get metrics
        metrics = churn.train_initial_model(df, target_variable, test_size, random_state)

        # Generate graphs
        graphs = churn.generate_visualizations(df, target_variable)


    return render_template('churnpred.html', metrics=metrics, graphs=graphs, form=form)



@main_bp.route('/predict_churn', methods=['POST'])
def predict_churn():
    """Predicts churn for new data provided in JSON format."""
    if not churn.model:
        return jsonify({"error": "Model is not trained yet. Train it first!"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    try:
        df = pd.DataFrame(data)
        predictions = churn.predict_churn(df)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500