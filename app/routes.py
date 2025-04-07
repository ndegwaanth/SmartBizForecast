from .forms import Registration, LoginForm, DynamicForm, ForgotPasswordForm, ResetPasswordForm
from flask_bcrypt import Bcrypt
# from flask_mysqldb import MySQL
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
from app.descriptive import Descriptive
from datetime import datetime
from flask import session
import numpy as np
from sklearn.metrics import mean_squared_error
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired
from werkzeug.security import generate_password_hash
from . import s

mail = Mail()


load_dotenv()
s = URLSafeTimedSerializer(os.getenv('SECRET_KEY'))

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
    session.clear()
    logout_user()
    flash("You have been logged out successfully.")
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


# @main_bp.route('/login', methods=['POST', 'GET'])
# def login():
# from . import collection


@main_bp.route('/login', methods=['POST', 'GET'])
def login():
    from . import collection

    if current_user.is_authenticated:
        return redirect(url_for('main.homepage'))

    form = LoginForm()
    if form.validate_on_submit():
        user_dict = collection.find_one({"Email": form.email.data})
        if user_dict and bcrypt.check_password_hash(user_dict["Password"], form.password.data):
            user = User(user_dict)
            login_user(user, remember=True)  # Add remember=True
            
            # Set session variables
            session['user_id'] = str(user_dict['_id'])
            session['logged_in'] = True
            session.permanent = True  # Make session persistent
            session.modified = True   # Mark session as modified
            
            flash('Login successful!', 'success')
            return redirect(url_for('main.homepage'))
    
    return render_template('login.html', form=form)

from flask_mail import Message
from itsdangerous import URLSafeTimedSerializer
import os
import logging
from datetime import datetime, timedelta


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



@main_bp.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    from . import collection  # Import your MongoDB collection
    from .forms import ForgotPasswordForm  # Import your form
    
    form = ForgotPasswordForm()
    
    if request.method == 'POST' and form.validate_on_submit():
        email = form.email.data.lower().strip()  # Normalize email
        user = collection.find_one({'Email': email})
        
        if not user:
            # Don't reveal whether email exists for security
            logger.info(f"Password reset requested for non-existent email: {email}")
            flash('If that email exists in our system, you will receive a password reset link.', 'info')
            return redirect(url_for('main.login'))
        
        # Check if a reset was recently requested
        # if user.get('reset_requested_at'):
        #     last_request = user['reset_requested_at']
        #     if datetime.utcnow() - last_request < timedelta(minutes=5):
        #         flash('A password reset link was already sent recently. Please check your email or wait 5 minutes.', 'warning')
        #         return redirect(url_for('main.login'))
        
        try:
            # Generate token with expiration (30 minutes)
            token = s.dumps(email, salt='password-reset-salt')
            link = url_for('main.reset_password', token=token, _external=True)
            
            # Update user with reset timestamp
            collection.update_one(
                {'Email': email},
                {'$set': {'reset_requested_at': datetime.utcnow()}}
            )
            
            # Create and send email
            msg = Message(
                'Password Reset Request',
                sender=os.getenv('MAIL_DEFAULT_SENDER', 'ndegwaanthony300@gmail.com'),
                recipients=[email]
            )
            
            msg.body = f"""Hello,
            
You requested a password reset for your account. Please click the following link to reset your password:
            
{link}
            
This link will expire in 30 minutes. If you didn't request this, please ignore this email.
            
Thank you,
The Support Team
            """
            
            # HTML version
            msg.html = f"""<html>
<body>
<p>Hello,</p>
<p>You requested a password reset for your account. Please click the following link to reset your password:</p>
<p><a href="{link}">{link}</a></p>
<p>This link will expire in 30 minutes. If you didn't request this, please ignore this email.</p>
<p>Thank you,<br>The Support Team</p>
</body>
</html>"""
            
            mail.send(msg)
            logger.info(f"Password reset email sent to {email}")
            flash('A password reset link has been sent to your email.', 'success')
            return redirect(url_for('main.login'))
            
        except Exception as e:
            logger.error(f"Failed to send password reset email to {email}: {str(e)}")
            flash('Failed to send password reset email. Please try again later.', 'danger')
    
    return render_template('forgot_pass.html', form=form)

@main_bp.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    from . import collection
    from .forms import ResetPasswordForm
    
    form = ResetPasswordForm()
    
    try:
        # Token expires after 30 minutes
        email = s.loads(token, salt='password-reset-salt', max_age=1800)
    except:
        flash('The reset link is invalid or has expired.', 'danger')
        return redirect(url_for('main.forgot_password'))
    
    if form.validate_on_submit():
        try:
            # Update password in database
            new_password = form.password.data
            hashed_password = generate_password_hash(new_password)
            
            collection.update_one(
                {'Email': email},
                {'$set': {
                    'Password': hashed_password,
                    'reset_requested_at': None  # Clear reset timestamp
                }}
            )
            
            flash('Your password has been updated successfully!', 'success')
            return redirect(url_for('main.login'))
        except Exception as e:
            logger.error(f"Failed to reset password for {email}: {str(e)}")
            flash('An error occurred while resetting your password. Please try again.', 'danger')
    
    return render_template('reset_password.html', form=form, token=token)


# @main_bp.route('/reset_password/<token>', methods=['GET', 'POST'])
# def reset_password(token):
#     from . import collection

#     form = ResetPasswordForm()
#     try:
#         email = s.loads(token, salt='password-reset-salt', max_age=3600)
#     except SignatureExpired:
#         flash('The reset link has expired.', 'danger')
#         return redirect(url_for('main.forgot_password'))

#     if request.method == 'POST' and form.validate_on_submit():
#         password = form.password.data
#         hashed_pass = bcrypt.generate_password_hash(password).decode('utf-8')
#         collection.update_one({'Email': email}, {'$set': {'password': hashed_pass}})
#         flash('Your password has been updated', 'success')
#         return redirect(url_for('main.login'))

#     return render_template('reset_pass.html', form=form)


# Directory where I will store the uploaded files
UPLOAD_FOLDER = 'data/user'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

# Ensuring the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @main_bp.route('/upload', methods=['POST', 'GET'])
# @login_required
# def upload_data():
#     if request.method == 'POST':
#         file = request.files.get('file')

#         # Check if a file is selected
#         if not file:
#             flash("No file was uploaded.", "error")
#             print("DEBUG: No file received in request")
#             return redirect(url_for('main.upload_data'))

#         # Check if the file is empty or has an invalid format
#         if file.filename == '':
#             flash("No selected file.", "error")
#             print("DEBUG: Empty filename")
#             return redirect(url_for('main.upload_data'))

#         if not allowed_file(file.filename):
#             flash("Invalid file format. Please upload a .csv or .xlsx file.", "error")
#             print(f"DEBUG: Invalid file type -> {file.filename}")
#             return redirect(url_for('main.upload_data'))

#         # Save file with a unique name
#         filename = secure_filename(file.filename)
#         unique_filename = f"{uuid.uuid4()}_{filename}"
#         file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
#         file.save(file_path)

#         try:
#             # Read the dataset (Only first 10 rows to avoid session overload)
#             if filename.lower().endswith('.csv'):
#                 df = pd.read_csv(file_path, nrows=10)
#             elif filename.lower().endswith('.xlsx'):
#                 df = pd.read_excel(file_path, nrows=10)
#             else:
#                 flash("Unsupported file format.", "error")
#                 print("DEBUG: File format not supported")
#                 return redirect(url_for('main.upload_data'))

#             print("DEBUG: File uploaded successfully:", file_path)
#             print("DEBUG: Data preview:\n", df.head())

#             # Store only column names in session, not the full dataset
#             session['columns'] = df.columns.tolist()
#             session['uploaded_file'] = unique_filename
#             session['dataset_preview'] = df.head(5).to_json()  # Store a preview, not full data

#             # Generate dynamic form
#             form = DynamicForm(columns=df.columns.tolist())

#             return render_template('prediction.html', headers=df.columns.tolist(), rows=df.values.tolist(), form=form)

#         except Exception as e:
#             flash(f"Error processing file: {str(e)}", "error")
#             print("DEBUG: Exception ->", str(e))
#             return redirect(url_for('main.upload_data'))

#     return render_template('homepage.html')



@main_bp.route('/model/training/configuration')
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
from .predictions import SalesPrediction
churn = ChurnPrediction()
sales = SalesPrediction()

@main_bp.route('/model/prediction/results/', methods=['GET', 'POST'])
def model_training():
    columns = session.get('columns', [])  # Retrieve column names from session
    form = DynamicForm(columns=columns)
    metrics = None
    graphs = []  # Initialize empty graph list
    model_summary = None

    if request.method == 'POST' and form.validate_on_submit():
        target_variable = form.target_variable.data
        predictor_variables = form.predictor_variables.data
        model_select = form.model_preferences.data
        test_size = float(form.test_size.data or 0.2)  # Default 20% test size
        random_state = int(form.random_state.data or 42)  # Default seed

        print(session.get('uploaded_file'))
        # Ensure dataset exists
        uploaded_file = session.get('uploaded_file')
        if not uploaded_file:
            flash("No dataset found. Please upload a dataset first.", "warning")
            print("No dataset found")
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
                print("Unsuported formatof the dataset")
                return redirect(url_for('main.upload_data'))
        except Exception as e:
            flash(f"Error loading dataset: {str(e)}", "error")
            print("Error in the dataset")
            return redirect(url_for('main.upload_data'))

        # Validate selected columns
        missing_columns = [col for col in predictor_variables if col not in df.columns]
        if target_variable not in df.columns or missing_columns:
            flash(f"Invalid columns: {missing_columns} missing.", "danger")
            print("Invalid data columns")
            return redirect(url_for('main.upload_data'))

        if model_select == 'logistic_regression':
            try:
                metrics = churn.train_initial_model(df, target_variable, test_size, random_state)
                X, y = churn.preprocess_data(df, target_variable)
                X = churn.scaler.transform(X)
                y_pred = churn.model.predict(X)
                graphs = churn.generate_visualizations(df, target_variable, y_pred, predictor_variables)
                
                # Debug prints to check what's being returned
                print("Metrics:", metrics)
                print("Graphs:", graphs)
                
                flash("Model training and visualization completed successfully.", "success")
                return render_template('churnpred.html', 
                                    metrics=metrics, 
                                    graphs=graphs, 
                                    form=form)
            except Exception as e:
                flash(f"Error during model training: {str(e)}", "error")
                return redirect(url_for('main.model_training', model_select=model_select))
        elif model_select == 'ARIMA':
            try:
                df[target_variable] = pd.to_numeric(df[target_variable], errors='coerce')
                df.dropna(subset=[target_variable], inplace=True)

                model = sales.train_initial_model(df, target_variable, test_size, random_state)
                flash("ARIMA model training completed successfully.", "success")

                # Get ARIMA predictions
                forecast = model.forecast(steps=len(df))

                # Evaluate model performance
                rmse = np.sqrt(mean_squared_error(df[target_variable].iloc[-len(forecast):], forecast))
                aic = model.aic
                bic = model.bic

                metrics = {
                    "RMSE": rmse,
                    "AIC": aic,
                    "BIC": bic
                }

                # Get model summary
                model_summary = model.summary().as_text()
                flash(f"Model Metrics - RMSE: {rmse:.4f}, AIC: {aic:.4f}, BIC: {bic:.4f}", "info")

                # Generate graphs
                graphs = sales.generate_visualizations(df, target_variable, forecast)
                flash("Graphs generated successfully.", "success")

            except Exception as e:
                flash(f"Error during ARIMA model training: {str(e)}", "error")
                return redirect(url_for('main.model_training'))

            return render_template('sales.html', metrics=metrics, graphs=graphs, form=form, model_summary=model_summary, model_select=model_select)

        else:
            flash('The selected model has not been implemented yet.', 'warning')
            return redirect(url_for('main.model_training'))
    
    # Add this return statement for GET requests
    return render_template('prediction.html', form=form, headers=columns, rows=[])


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
    

@main_bp.route('/descriptive/statistics/')
def descriptive():
    # Ensure dataset exists
    uploaded_file = session.get('uploaded_file')
    if not uploaded_file:
        flash("No dataset found. Please upload a dataset first.", "warning")
        return redirect(url_for('main.upload_data'))

    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file)

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        flash("Upload required data format", "danger")
        return redirect(url_for('main.upload_data'))

    # Generate descriptive statistics using Descriptive class
    stats = Descriptive.generate_descriptive_stats(df)

    return render_template('descriptive.html', stats=stats)


# Add these near your other imports
from bson import ObjectId
import json
from datetime import datetime

@main_bp.route('/api/recent_files')
@login_required
def recent_files():
    from . import collection

    """Return list of recently uploaded files for the current user"""
    try:
        # Get user's recent files (last 5)
        recent_files = list(collection.find(
            {'user_id': ObjectId(session['user_id'])},
            {'filename': 1, 'upload_date': 1, '_id': 0}
        ).sort('upload_date', -1).limit(5))
        
        # Convert ObjectId and datetime to strings
        for file in recent_files:
            file['upload_date'] = file['upload_date'].isoformat()
        
        return jsonify({
            "status": "success",
            "files": recent_files
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@main_bp.route('/api/dataset_stats')
@login_required
def dataset_stats():
    """Return statistics about the current dataset"""
    try:
        if 'uploaded_file' not in session:
            return jsonify({
                "status": "error",
                "message": "No dataset loaded"
            }), 400
        
        file_path = os.path.join(UPLOAD_FOLDER, session['uploaded_file'])
        
        # Read the file
        if session['uploaded_file'].endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Calculate stats
        numeric_cols = len(df.select_dtypes(include=['number']).columns)
        categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        
        return jsonify({
            "status": "success",
            "stats": {
                "row_count": len(df),
                "column_count": len(df.columns),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024)  # in MB
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@main_bp.route('/api/visualizations')
@login_required
def visualizations():
    """Generate basic visualizations for the current dataset"""
    try:
        if 'uploaded_file' not in session:
            return jsonify({
                "status": "error",
                "message": "No dataset loaded"
            }), 400
        
        file_path = os.path.join(UPLOAD_FOLDER, session['uploaded_file'])
        
        # Read the file
        if session['uploaded_file'].endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        visualizations = []
        
        # Generate histogram for first numeric column
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            hist, bins = np.histogram(df[col].dropna(), bins=10)
            visualizations.append({
                "id": "histogram",
                "type": "chart",
                "title": f"Distribution of {col}",
                "chartType": "bar",
                "labels": [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(hist))],
                "datasets": [{
                    "label": "Frequency",
                    "data": hist.tolist(),
                    "backgroundColor": "rgba(54, 162, 235, 0.7)"
                }]
            })
        
        # Generate bar chart for first categorical column
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            col = categorical_cols[0]
            value_counts = df[col].value_counts().head(10)
            visualizations.append({
                "id": "barchart",
                "type": "chart",
                "title": f"Top {col} Values",
                "chartType": "bar",
                "labels": value_counts.index.tolist(),
                "datasets": [{
                    "label": "Count",
                    "data": value_counts.values.tolist(),
                    "backgroundColor": "rgba(75, 192, 192, 0.7)"
                }]
            })
        
        return jsonify({
            "status": "success",
            "visualizations": visualizations
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@main_bp.route('/api/data_preview')
@login_required
def data_preview():
    """Return a preview of the current dataset"""
    try:
        if 'uploaded_file' not in session:
            return jsonify({
                "status": "error",
                "message": "No dataset loaded"
            }), 400
        
        file_path = os.path.join(UPLOAD_FOLDER, session['uploaded_file'])
        
        # Read the file (first 10 rows only)
        if session['uploaded_file'].endswith('.csv'):
            df = pd.read_csv(file_path, nrows=10)
        else:
            df = pd.read_excel(file_path, nrows=10)
        
        # Convert NaN to None for JSON serialization
        df = df.where(pd.notnull(df), None)
        
        return jsonify({
            "status": "success",
            "headers": df.columns.tolist(),
            "rows": df.values.tolist()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Modify your existing upload_data route to track files
@main_bp.route('/upload', methods=['POST', 'GET'])
@login_required
def upload_data():
    from . import collection
    if request.method == 'POST':
        file = request.files.get('file')

        if not file:
            flash("No file was uploaded.", "error")
            return redirect(url_for('main.upload_data'))

        if file.filename == '':
            flash("No selected file.", "error")
            return redirect(url_for('main.upload_data'))

        if not allowed_file(file.filename):
            flash("Invalid file format. Please upload a .csv or .xlsx file.", "error")
            return redirect(url_for('main.upload_data'))

        # Save file with unique name
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)

        try:
            # Read the dataset (first 10 rows)
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(file_path, nrows=10)
            else:
                df = pd.read_excel(file_path, nrows=10)

            # Store file info in database
            file_data = {
                "user_id": ObjectId(session['user_id']),
                "filename": filename,
                "stored_filename": unique_filename,
                "upload_date": datetime.utcnow(),
                "size": os.path.getsize(file_path),
                "columns": df.columns.tolist()
            }
            collection.insert_one(file_data)

            # Update session
            session['columns'] = df.columns.tolist()
            session['uploaded_file'] = unique_filename
            session['dataset_preview'] = df.head(5).to_json()

            flash("File uploaded successfully!", "success")
            return redirect(url_for('main.homepage'))

        except Exception as e:
            flash(f"Error processing file: {str(e)}", "error")
            return redirect(url_for('main.upload_data'))

    return render_template('homepage.html')


@main_bp.route('/api/column_stats/<column_name>')
@login_required
def column_stats(column_name):
    try:
        if 'uploaded_file' not in session:
            return jsonify({
                "error": "No dataset loaded. Please upload data first.",
                "status": "error"
            }), 400
        
        file_path = os.path.join(UPLOAD_FOLDER, session['uploaded_file'])
        
        # Read the file
        try:
            if session['uploaded_file'].endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
        except Exception as e:
            return jsonify({
                "error": f"Failed to read dataset: {str(e)}",
                "status": "error"
            }), 500

        if column_name not in df.columns:
            return jsonify({
                "error": f"Column '{column_name}' not found in dataset",
                "status": "error"
            }), 404
        
        col = df[column_name]
        stats = {
            "name": column_name,
            "dtype": str(col.dtype),
            "count": len(col),
            "missing": col.isna().sum(),
            "unique": col.nunique()
        }
        
        if pd.api.types.is_numeric_dtype(col):
            stats.update({
                "type": "numeric",
                "min": float(col.min()),
                "max": float(col.max()),
                "mean": float(col.mean()),
                "median": float(col.median()),
                "std": float(col.std()),
                "histogram": {
                    "values": pd.cut(col, bins=10).value_counts().sort_index().to_dict(),
                    "range": [float(col.min()), float(col.max())]
                }
            })
        else:
            stats.update({
                "type": "categorical",
                "top_values": col.value_counts().head(5).to_dict()
            })
            
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500