from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    profile_info = db.Column(db.String(200), nullable=True)
    def set_password(self, password):
        """Hashes the password and stores it."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Checks the hashed password against the provided one."""
        return check_password_hash(self.password_hash, password)

class User(UserMixin):
    def __init__(self, user_dict):
        # The user_dict is the dictionary returned from MongoDB for the user
        self.id = str(user_dict['_id'])  # User ID stored as string
        self.username = user_dict['Username']
        self.email = user_dict['Email']
        self.password_hash = user_dict['Password']  # Make sure this is a hashed password
    
    # Flask-Login requires this method to return a string that uniquely identifies this user
    def get_id(self):
        return self.id
