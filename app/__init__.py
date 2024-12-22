from flask import Flask
from dotenv import load_dotenv
from flask_bcrypt import Bcrypt
import secrets
from flask_wtf.csrf import CSRFProtect
import os
from flask_login import LoginManager
# import yaml
from flask_mysqldb import MySQL
import os

# Load environment variables
load_dotenv()
# db = yaml.load(open('db.yaml'), Loader=yaml.SafeLoader)

# Create Flask app
app = Flask(__name__)

# Set a secret key for sessions and CSRF
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))

# Set MySQL database configuration
app.config['MYSQL_HOST'] = os.getenv('mysql_host') 
app.config['MYSQL_USER'] = os.getenv('mysql_user')
app.config['MYSQL_PASSWORD'] = os.getenv('mysql_password')
app.config['MYSQL_DB'] = os.getenv('mysql_db')

mysql = MySQL(app)

# Initialize extensions
bcrypt = Bcrypt(app)
csrf = CSRFProtect(app)

# Login Manager
# login_manager = LoginManager(app)
# login_manager.login_view = 'main.login'


# Import and register blueprints
from .routes import main_bp
app.register_blueprint(main_bp)
