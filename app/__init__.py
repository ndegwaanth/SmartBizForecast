from flask import Flask
from dotenv import load_dotenv
from flask_bcrypt import Bcrypt
import secrets
from flask_wtf.csrf import CSRFProtect
import os

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)

# Set a secret key for sessions and CSRF
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))

# Initialize extensions
bcrypt = Bcrypt(app)
csrf = CSRFProtect(app)

# Import and register blueprints
from .routes import main_bp
app.register_blueprint(main_bp)
