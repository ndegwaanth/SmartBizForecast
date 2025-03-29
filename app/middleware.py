from flask import session, redirect, url_for
from functools import wraps

def check_valid_user(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('main.login'))
        return f(*args, **kwargs)
    return decorated_function