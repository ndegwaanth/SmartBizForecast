from flask import session, render_template, redirect, request, Blueprint, url_for
from .forms import Registration, LoginForm
from flask_bcrypt import Bcrypt
from flask_mysqldb import MySQL
import secrets
from . import mysql

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