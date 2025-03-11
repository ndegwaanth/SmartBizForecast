from flask import Flask
from dotenv import load_dotenv
from flask_bcrypt import Bcrypt
import secrets
from flask_wtf.csrf import CSRFProtect
import os
from flask_login import LoginManager
from pymongo import MongoClient
from flask_session import Session

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__, template_folder="templates")

# Ensure SECRET_KEY is always set
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))  # Generates a secure key if missing

# Initialize CSRF protection
csrf = CSRFProtect(app)

# MongoDB Configuration
Mongo_url = os.getenv("MONGODB_URL")
client = MongoClient(Mongo_url)
db = client['Users']
collection = db['users-info']

# Initialize other extensions
bcrypt = Bcrypt(app)

# Initialize Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = 'main.login'

# Flask-Login User Loader
from .models import User

@login_manager.user_loader
def load_user(user_id):
    # Retrieve user from MongoDB by their ID
    user_dict = collection.find_one({"_id": user_id})
    if user_dict:
        return User(user_dict)
    return None

# Import and register blueprints
from .routes import main_bp
app.register_blueprint(main_bp)