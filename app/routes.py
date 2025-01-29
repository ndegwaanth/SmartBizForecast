from .forms import Registration, LoginForm, ModelForm
from flask_bcrypt import Bcrypt
from flask_mysqldb import MySQL
import secrets
from . import mysql
from flask import session, render_template, redirect, request, Blueprint, url_for, flash
import pandas as pd
from werkzeug.utils import secure_filename
import os


key = secrets.token_hex(64)

main_bp = Blueprint('main', __name__)
bcrypt = Bcrypt()


@main_bp.before_app_request
def initialize():
    print("Checking MySQL connection...")
    if mysql.connection is None:
        print("MySQL connection is not established.")
    else:
        print("MySQL connection is established.")
@main_bp.route('/')
def landing_page():
    return render_template('landing_page.html')


@main_bp.route("/signup", methods=["POST", "GET"])
def register():
    form = Registration()
    if form.validate_on_submit() and request.method == 'POST':
        from . import mysql

        firstname = form.firstname.data
        lastname = form.lastname.data
        username_reg = form.username.data
        email_reg = form.email.data
        password_reg = form.password.data
        confirm_password = form.confirm_password.data

        if password_reg != confirm_password:
            return render_template("register.html", form=form, error="Passwords do not match")

        password_hash = bcrypt.generate_password_hash(password_reg).decode('utf-8')
        print('Connection Initializing...')

        try:
            cur = mysql.connection.cursor()
            cur.execute(
                "INSERT INTO User (firstname, lastname, username, email, password) VALUES (%s, %s, %s, %s, %s)",
                (firstname, lastname, username_reg, email_reg, password_hash),
            )
            mysql.connection.commit()
            cur.close()
            print("Connection is Successfull")
            return redirect(url_for("main.homepage"))
        except Exception as e:
            print(f'Connection failled: {e}')
            return redirect(url_for("main.homepage"))
            # return render_template("register.html", form=form, error="Database error: " + str(e))
    return render_template("register.html", form=form)


@main_bp.route('/login', methods=['POST', 'GET'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        from . import mysql

        email = form.email.data
        password = form.password.data

        print('Connection Initializing...')
        try:
            cur = mysql.connection.cursor()
            cur.execute("SELECT password FROM User WHERE email = %s", (email,))
            result = cur.fetchone()
            cur.close()

            if result and bcrypt.check_password_hash(result[0], password):
                session['user_email'] = email
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
            if filename.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif filename.endswith('.xlsx'):
                data = pd.read_excel(file_path)

            # Store column names in session
            session['uploaded_data'] = {"columns": data.columns.tolist()}

            flash("File uploaded successfully!", "success")
            return redirect(url_for('main.predictions'))
        else:
            flash("Invalid file format. Please upload a .csv or .xlsx file.", "error")
            return redirect(url_for('main.upload_data'))

    return render_template('homepage.html')


@main_bp.route("/Predictions", methods=['GET', 'POST'])
def predictions():
    form = ModelForm()
    
    # Check if data has been uploaded (stored in session or elsewhere)
    uploaded_data = session.get("uploaded_data")  # Assume data is stored in session
    if uploaded_data:
        columns = uploaded_data["columns"]  # Example structure: {"columns": ["col1", "col2", ...]}
        form.target.choices = [(col, col) for col in columns]
        form.predictors.choices = [(col, col) for col in columns]
    else:
        form.target.choices = []
        form.predictors.choices = []
    
    if form.validate_on_submit():
        target = form.target.data
        predictors = form.predictors.data
        model_selection = form.model_selection.data
        # Handle prediction logic here
        
        flash(f"Target: {target}, Predictors: {predictors}, Model: {model_selection}")
        return redirect(url_for("main.predictions"))

    return render_template('prediction.html', form=form)
