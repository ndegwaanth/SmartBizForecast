from .forms import Registration, LoginForm
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
from flask_login import login_user, logout_user, current_user
# from werkzeug.security import generate_password_hash, check_password_hash

load_dotenv()


key = secrets.token_hex(64)

main_bp = Blueprint('main', __name__)
bcrypt = Bcrypt()



@main_bp.route('/')
def landing_page():
    return render_template('landing_page.html')


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
                return redirect(url_for('main.landing_page'))
            else:
                return render_template("login.html", form=form, error="Invalid email or password")
        except Exception as e:
            return render_template("login.html", form=form, error="Database error: " + str(e))
    return render_template("login.html", form=form)

@main_bp.route('/homepage')
def homepage():
    return render_template('homepage.html')


# Directory where you will store the uploaded files
UPLOAD_FOLDER = 'data/user'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

# Ensure the upload folder exists
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
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Process the file
            try:
                if filename.endswith('.csv'):
                    data = pd.read_csv(file_path)
                elif filename.endswith('.xlsx'):
                    data = pd.read_excel(file_path)
                else:
                    flash("Unsupported file format.", "error")
                    return redirect(url_for('main.upload_data'))

                # Automate data cleaning
                data = data_cleaning(data)

                # Store column names in session
                session['uploaded_data'] = {"columns": data.columns.tolist()}

                flash("File uploaded and cleaned successfully!", "success")
                return redirect(url_for('main.predictions'))

            except Exception as e:
                flash(f"Error processing file: {str(e)}", "error")
                return redirect(url_for('main.upload_data'))
        else:
            flash("Invalid file format. Please upload a .csv or .xlsx file.", "error")
            return redirect(url_for('main.upload_data'))

    return render_template('homepage.html')
