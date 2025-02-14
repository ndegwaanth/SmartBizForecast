from app import app
from app.routes import run_streamlit


if __name__ == "__main__":
    run_streamlit()
    app.run(debug=True)
