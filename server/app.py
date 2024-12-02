from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import pickle
import pandas as pd
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the models
with open('models/pkl/logistic_regression.pkl', 'rb') as f:
    logistic_regression_model = pickle.load(f)

with open('models/pkl/random_forest.pkl', 'rb') as f:
    random_forest_model = pickle.load(f)

with open('models/pkl/decision_tree_model.pkl', 'rb') as f:
    decision_tree_model = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/predict/log', methods=['POST'])
def predict_logistic_regression():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])

    # Predict using logistic regression model
    prediction = logistic_regression_model.predict(df)
    result = prediction[0]

    message = "Your bike may be recovered" if result == 1 else "Your bike may NOT be recovered"
    return jsonify({'logistic_regression_prediction': message})

@app.route('/predict/rf', methods=['POST'])
def predict_random_forest():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])

    # Predict using random forest model
    prediction = random_forest_model.predict(df)
    result = prediction[0]

    message = "Your bike may be recovered" if result == 1 else "Your bike may NOT be recovered"
    return jsonify({'random_forest_prediction': message})

@app.route('/predict/dt', methods=['POST'])
def predict_decision_tree():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])

    # Predict using decision tree model
    prediction = decision_tree_model.predict(df)
    result = prediction[0]

    message = "Your bike may be recovered" if result == 1 else "Your bike may NOT be recovered"
    return jsonify({'decision_tree_prediction': message})

if __name__ == '__main__':
    app.run(debug=True)