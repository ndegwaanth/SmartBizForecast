from flask import Flask
from dotenv import load_dotenv
from flask_bcrypt import Bcrypt
import secrets
from flask_wtf.csrf import CSRFProtect
import os

load_dotenv()


app = Flask(__name__)

bcrypy = Bcrypt(app)
session_secrete = secrets.token_hex(32)

csrf = CSRFProtect(app)

from .routes import main_bp

app.register_blueprint(main_bp)