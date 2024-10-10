#!/bin/bash

# Defining the project name
PROJECT_NAME="Predictive_Analysis"

# Creating the main project directory
mkdir -p $PROJECT_NAME

# Creating app directory and files
mkdir -p $PROJECT_NAME/app/static
mkdir -p $PROJECT_NAME/app/templates

touch $PROJECT_NAME/app/__init__.py
touch $PROJECT_NAME/app/routes.py
touch $PROJECT_NAME/app/models.py
touch $PROJECT_NAME/app/tasks.py
touch $PROJECT_NAME/app/static/.gitkeep
touch $PROJECT_NAME/app/templates/index.html
touch $PROJECT_NAME/app/templates/dashboard.html
touch $PROJECT_NAME/app/templates/prediction.html
touch $PROJECT_NAME/app/utils.py

# Creating data directory and example dataset
mkdir -p $PROJECT_NAME/data
touch $PROJECT_NAME/data/example.csv

# Creating tests directory and test file
mkdir -p $PROJECT_NAME/tests
touch $PROJECT_NAME/tests/test_app.py

# Creating instance directory and config file
mkdir -p $PROJECT_NAME/instance
touch $PROJECT_NAME/instance/config.py

# Creating project root files
touch $PROJECT_NAME/.env
touch $PROJECT_NAME/requirements.txt
touch $PROJECT_NAME/manage.py
touch $PROJECT_NAME/celery_worker.py
touch $PROJECT_NAME/README.md
touch $PROJECT_NAME/run.py

echo "Project structure for $PROJECT_NAME has been created successfully."
