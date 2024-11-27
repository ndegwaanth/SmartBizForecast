from flask import Flask, session, render_template, redirect, request, Blueprint, url_for
from .forms import Registration, LoginForm
from flask_bcrypt import Bcrypt
from flask_wtf import CSRFProtect
import secrets

main_bp = Blueprint('main', __name__)
bcrypt = Bcrypt()


@main_bp.route('/')
def homepage():
    return render_template('dashboard.html')


@main_bp.route("/signup", methods=["POST", "GET"])
def register():
    form = Registration()
    if form.validate_on_submit():
        username_reg = form.username.data
        email_reg = form.email.data
        password_reg = form.password.data
        confirm_password = form.confirm_password.data
        firstname = form.firstname.data
        lastname = form.lastname.data
        
        if password_reg != confirm_password:
            return "The passwords do not match"
        
        password_reg_ecrpt = bcrypt.generate_password_hash(password_reg).decode('utf8')

        # Here you would typically save the user to the database
        # Example: User.create(username=username_reg, email=email_reg, password=password_reg_ecrpt)

        return render_template("dashboard.html")
    else:
        err = form.errors
        print(err)
        return render_template("register.html", form=form)


@main_bp.route('/login', methods=['POST', 'GET'])
def login():
    form = LoginForm()

    # Simulating stored user credentials (use a database in real applications)
    registered_email = "ndegwaanthony300@gmail.com"
    registered_password_hash = bcrypt.generate_password_hash("password123").decode("utf-8")

    if form.validate_on_submit():
        email_log = form.email.data
        password_log = form.password.data

        # Validate email and password
        if email_log == registered_email and bcrypt.check_password_hash(registered_password_hash, password_log):
            # Successful login
            session['user_email'] = email_log
            return redirect(url_for('main.homepage'))
        else:
            # Invalid credentials
            return render_template("login.html", form=form, error="Invalid email or password")

    return render_template("login.html", form=form)
