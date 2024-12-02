# Bike Recovery Predictor

## Team Members (Group 2)
- Attidiye Dilshan Nayanamadhu Liyanage
- Deanne Laylay
- Haneef Muhammad Osaina Syed Hussain
- Ninghua Zhang
- Wenjie Zhou

## Description
The Bike Recovery Predictor is a Python Flask application that predicts the likelihood of a bike being recovered based on various factors using logistic regression, random forest, and decision tree models.

## Deployment
- Deployed on Render [https://bikerecoverypredictor.onrender.com/](https://bikerecoverypredictor.onrender.com/)

## Features
- **Bike Recovery Prediction**: Predicts the likelihood of a bike being recovered using logistic regression, random forest, and decision tree models.

## Tech Stack
- Python Flask
- scikit-learn
- Gunicorn
- Bootstrap (for frontend styling)

## Installation & Setup
1. Navigate to the `server` directory:
    ```bash
    cd server
    ```
2. Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # On Windows
    # source .venv/bin/activate  # On macOS/Linux
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Environment Variables
None

## Running the Application
1. Navigate to the `server` directory and activate the virtual environment:
    ```bash
    cd server
    .venv\Scripts\activate  # On Windows
    # source .venv/bin/activate  # On macOS/Linux
    ```
2. Run the Flask application:
    ```bash
    python app.py
    ```

## API Endpoints
- **POST** `/predict/log` - Predicts bike recovery using the logistic regression model.
- **POST** `/predict/rf` - Predicts bike recovery using the random forest model.
- **POST** `/predict/dt` - Predicts bike recovery using the decision tree model.

## Frontend
The frontend is built using Bootstrap and includes a form to input the necessary data for prediction. The results are displayed on the same page.

## Usage
1. Open the application in your web browser.
2. Fill in the form with the required data.
3. Click on the "Predict with Logistic Regression", "Predict with Random Forest", or "Predict with Decision Tree" button to get the prediction.
4. The prediction results will be displayed below the form.