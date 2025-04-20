from flask import Flask, flash
from dotenv import load_dotenv
from flask_bcrypt import Bcrypt
import secrets
from flask_wtf.csrf import CSRFProtect
import os
from flask_login import LoginManager
from pymongo import MongoClient
from flask_session import Session
from datetime import timedelta
from .models import User
from flask_mail import Mail
from itsdangerous import URLSafeTimedSerializer
from flask_cors import CORS
from flask import session, render_template, redirect, request, Blueprint, url_for, flash


# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__, template_folder="templates")

# Essential configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))
s = URLSafeTimedSerializer(app.config['SECRET_KEY'])

CORS(app)

# Flask-Session configuration
app.config.update({
    'SESSION_TYPE': 'filesystem',
    'SESSION_FILE_DIR': './flask_sessions',
    'SESSION_PERMANENT': True,
    'PERMANENT_SESSION_LIFETIME': timedelta(hours=1),
    'SESSION_COOKIE_NAME': 'pa_session',
    'SESSION_COOKIE_SECURE': False,  # True in production
    'SESSION_COOKIE_HTTPONLY': True,
    'SESSION_COOKIE_SAMESITE': 'Lax',
    'SESSION_REFRESH_EACH_REQUEST': True,
    'WTF_CSRF_TIME_LIMIT': 3600,  # 1 hour
    'WTF_CSRF_SSL_STRICT': False  # True in production
})

@app.before_request
def check_session():
    print(f"\n[SESSION DEBUG] Before Request - Session ID: {session.sid}")
    print(f"[SESSION DEBUG] Session exists: {bool(session)}")
    print(f"[SESSION DEBUG] CSRF Token: {session.get('_csrf_token')}")
    print(f"[SESSION DEBUG] Session keys: {list(session.keys())}\n")

@app.after_request
def add_session_header(response):
    response.headers['X-Session-ID'] = session.sid if session else 'None'
    return response

mail = Mail(app)


from flask_wtf.csrf import CSRFProtect, generate_csrf

# app creation and before blueprint registration
csrf = CSRFProtect(app)

@app.before_request
def ensure_csrf_token():
    """Ensure CSRF token is in session with correct key"""
    if '_csrf_token' not in session:
        # If we have the token under the wrong key, move it
        if 'csrf_token' in session:
            session['_csrf_token'] = session.pop('csrf_token')
        else:
            # Generate new token if none exists
            session['_csrf_token'] = generate_csrf()
    # Ensure session is saved
    if session.modified:
        session.permanent = True
        session.modified = True

# after_request handler
@app.after_request
def inject_csrf_token(response):
    """Ensure CSRF token cookie is set"""
    response.set_cookie(
        'csrf_token',
        session.get('_csrf_token', ''),
        secure=app.config.get('SESSION_COOKIE_SECURE', False),
        httponly=True,
        samesite='Lax'
    )
    return response


Session(app)
app.config['WTF_CSRF_TIME_LIMIT'] = 3600
app.config['WTF_CSRF_CHECK_DEFAULT'] = True


from flask_wtf.csrf import CSRFError

@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    flash('The form submission was invalid. Please try again.', 'danger')
    return redirect(request.referrer or url_for('main.homepage'))

# MongoDB Configuration
# In __init__.py
try:
    Mongo_url = os.getenv("MONGODB_URL")
    if not Mongo_url:
        raise ValueError("MONGODB_URL environment variable not set")
    client = MongoClient(Mongo_url, serverSelectionTimeoutMS=5000)  # 5 second timeout
    client.server_info()  # Test the connection
    db = client['Users']
    # collection = db['users-info']
    collection = db['users-data']
    print("Successfully connected to MongoDB")
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    collection = None  # This will help identify if DB isn't connected

bcrypt = Bcrypt(app)




app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')

s = URLSafeTimedSerializer(app.config['SECRET_KEY'])



# Flask-Login configuration
login_manager = LoginManager(app)
login_manager.login_view = 'main.login'
login_manager.session_protection = "strong"

@login_manager.user_loader
def load_user(user_id):
    try:
        # MongoDB uses ObjectId - ensure proper conversion
        from bson.objectid import ObjectId
        user_dict = collection.find_one({"_id": ObjectId(user_id)})
        if user_dict:
            return User(user_dict)
        return None
    except:
        return None

# Register blueprints
from .routes import main_bp
app.register_blueprint(main_bp)


# Create session directory if it doesn't exist
# At the end of __init__.py
if app.config['SESSION_TYPE'] == 'filesystem':
    os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
    print(f"Session files will be stored in: {app.config['SESSION_FILE_DIR']}")