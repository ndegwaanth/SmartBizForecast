import pandas as pd

# Create a list of 20 sample test cases
test_cases = [
    {"Test Case": "Login with valid credentials", "Input": "Correct email & password", "Expected Output": "User dashboard loads", "Actual Output": "User dashboard loads", "Status": "Pass"},
    {"Test Case": "Login with invalid password", "Input": "Correct email, wrong password", "Expected Output": "Authentication error", "Actual Output": "Authentication error", "Status": "Pass"},
    {"Test Case": "Registration with existing email", "Input": "Already used email", "Expected Output": "Error: Email already exists", "Actual Output": "Error displayed", "Status": "Pass"},
    {"Test Case": "Upload valid CSV file", "Input": "Valid sales_data.csv", "Expected Output": "File uploaded and previewed", "Actual Output": "File uploaded and previewed", "Status": "Pass"},
    {"Test Case": "Upload invalid file format", "Input": "sales_data.txt", "Expected Output": "Error: Invalid file format", "Actual Output": "Error message shown", "Status": "Pass"},
    {"Test Case": "Data cleaning with missing values", "Input": "Data with nulls", "Expected Output": "Null values removed or imputed", "Actual Output": "Nulls handled correctly", "Status": "Pass"},
    {"Test Case": "Navigation to dashboard", "Input": "Click on Dashboard link", "Expected Output": "Streamlit dashboard opens", "Actual Output": "Streamlit dashboard opens", "Status": "Pass"},
    {"Test Case": "Access without login", "Input": "Direct URL access", "Expected Output": "Redirect to login page", "Actual Output": "Redirected to login", "Status": "Pass"},
    {"Test Case": "Incorrect CSV structure", "Input": "CSV missing headers", "Expected Output": "Error: Invalid format", "Actual Output": "Error message shown", "Status": "Pass"},
    {"Test Case": "Prediction with cleaned data", "Input": "Clean data set", "Expected Output": "Return prediction results", "Actual Output": "Predictions generated", "Status": "Pass"},
    {"Test Case": "Descriptive stats page", "Input": "Valid uploaded data", "Expected Output": "Display summary statistics", "Actual Output": "Stats displayed", "Status": "Pass"},
    {"Test Case": "Missing value detection", "Input": "Dataset with missing values", "Expected Output": "Missing values flagged", "Actual Output": "Missing values flagged", "Status": "Pass"},
    {"Test Case": "Histogram plot generation", "Input": "Numerical column", "Expected Output": "Histogram displayed", "Actual Output": "Correct histogram", "Status": "Pass"},
    {"Test Case": "Correlation heatmap display", "Input": "Numeric dataset", "Expected Output": "Show heatmap", "Actual Output": "Heatmap shown", "Status": "Pass"},
    {"Test Case": "Churn prediction with valid input", "Input": "Customer data", "Expected Output": "Return churn probability", "Actual Output": "Churn prediction displayed", "Status": "Pass"},
    {"Test Case": "Sales prediction output", "Input": "Historical sales data", "Expected Output": "Forecasted values shown", "Actual Output": "Forecast shown", "Status": "Pass"},
    {"Test Case": "Dashboard filter interaction", "Input": "Select filter values", "Expected Output": "Update visuals", "Actual Output": "Dashboard updates", "Status": "Pass"},
    {"Test Case": "Mobile view UI", "Input": "Open on mobile browser", "Expected Output": "Responsive design renders", "Actual Output": "Mobile layout loaded", "Status": "Pass"},
    {"Test Case": "Server crash recovery", "Input": "Restart Flask", "Expected Output": "Resume previous state", "Actual Output": "System restarted successfully", "Status": "Pass"},
    {"Test Case": "Download cleaned dataset", "Input": "Click download button", "Expected Output": "File downloaded", "Actual Output": "File downloaded", "Status": "Pass"},
]

# Convert to DataFrame
df = pd.DataFrame(test_cases)

# Save to Excel
excel_path = "/mnt/data/System_Test_Cases.xlsx"
df.to_excel(excel_path, index=False)

excel_path

