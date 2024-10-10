from flask import Flask, session, render_template, redirect, request, Blueprint
from .forms import Registration


main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def homepage():
    return render_template('dashboard.html')


@main_bp.route('login')
def login():
    return render_template('login.html')