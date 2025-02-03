from flask import Flask
from dotenv import load_dotenv
from flask_bcrypt import Bcrypt
import secrets
from flask_wtf.csrf import CSRFProtect
import os
from flask_login import LoginManager
from pymongo import MongoClient
from flask_wtf import CSRFProtect

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)


#CSRF
csrf = CSRFProtect(app)

# Set a secret key for sessions and CSRF
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))

# MongoDB Configuration
Mongo_url = os.getenv("MONGODB_URL")
client = MongoClient(Mongo_url)
db = client['Users']
collection = db['users-info']

# Initialize extensions
bcrypt = Bcrypt(app)
csrf = CSRFProtect(app)

# Login Manager
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
