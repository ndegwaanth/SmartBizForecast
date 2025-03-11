from flask import Flask, render_template, jsonify, redirect


app = Flask(__name__)

data_dict = {
    "name": "John Doe",
    "age": 30,
    "email": "john.doe@example.com",
    "city": "New York",
    "country": "USA",
    "phone": "+1-555-1234",
    "is_student": False,
    "height_cm": 175,
    "weight_kg": 70,
    "hobbies": ["reading", "cycling", "gaming"],
    "skills": {"Python": "Advanced", "SQL": "Intermediate", "JavaScript": "Beginner"},
    "salary": 75000,
    "currency": "USD",
    "married": True,
    "children": 2,
    "pets": ["dog", "cat"],
    "car": {"brand": "Toyota", "model": "Camry", "year": 2020},
    "favorite_color": "Blue",
    "has_drivers_license": True,
    "subscription_plan": "Premium"
}


@app.route('/api/random/data')
def random():
    if data_dict:
        return jsonify(data_dict)



if __name__ == '__main__':
    app.run(debug=True, port=3000)
