from flask import Flask, request, jsonify
import pandas as pd
import os
from pycaret.regression import *
from pycaret.regression import setup, compare_models, save_model, predict_model
from pycaret.regression import pull, load_model
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import joblib

app = Flask(__name__)

# Set max file size (optional)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

# File upload endpoint
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)  # Save the file temporarily
        return jsonify({"message": "File uploaded successfully!", "file_path": file_path})
    except Exception as e:
        return jsonify({"error": f"Error processing the file: {e}"}), 500

# Model training endpoint
@app.route('/model', methods=['POST'])
def run_model():
    data = request.json
    if not data or "data" not in data or "target" not in data:
        return jsonify({"error": "Invalid request payload"}), 400
    try:
        df = pd.DataFrame(data["data"])
        target = data["target"]

        # Check if the target column exists in the dataset
        if target not in df.columns:
            return jsonify({"error": f"Target column '{target}' not found in dataset"}), 400

        # Check for missing values in the target column
        if df[target].isnull().any():
            return jsonify({"error": f"Target column '{target}' contains missing values"}), 400

        # Setup PyCaret (remove the 'silent' argument)
        setup(df, target=target, verbose=False)  # Use 'verbose=False' to suppress logs

        # Compare models and select top 5
        top_5_models = compare_models(n_select=5)

        # Get performance metrics for each model
        model_details = []
        for i, model in enumerate(top_5_models):
            model_name = str(model).split("(")[0]  # Extract model name
            metrics = pull()  # Get metrics for the current model
            model_details.append({
                "rank": i + 1,
                "model_name": model_name,
                "rmse": metrics.loc[0, "RMSE"],
                "mae": metrics.loc[0, "MAE"],
                "r2": metrics.loc[0, "R2"]
            })
            save_model(model, f"top_model_{i + 1}")  # Save each model

        return jsonify({
            "message": "Model training completed successfully!",
            "top_5_models": model_details
        })
    except Exception as e:
        # Log the full error message
        print(f"Error during model training: {str(e)}")
        return jsonify({"error": f"Error during model training: {str(e)}"}), 500
# Ensemble endpoint
@app.route('/ensemble', methods=['POST'])
def create_ensemble():
    try:
        # Load the dataset
        df = pd.read_csv('dataset.csv')
        target = df.columns[-1]  # Assuming the last column is the target

        # Setup PyCaret
        setup(df, target=target)

        # Compare models and select top 5
        top_5_models = compare_models(n_select=5)

        # Ensemble the top 3 models using stacking
        estimators = [(f'model_{i}', model) for i, model in enumerate(top_5_models[:3])]
        ensemble_model = StackingRegressor(estimators=estimators, final_estimator=Ridge())
        ensemble_model.fit(df.drop(columns=[target]), df[target])

        # Save the ensemble model
        save_model(ensemble_model, "ensemble_model")
        return jsonify({
            "message": "Ensemble model created successfully!",
            "ensemble_details": {"ensemble_model": str(ensemble_model)}
        })
    except Exception as e:
        return jsonify({"error": f"Error during ensemble creation: {e}"}), 500

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or "data" not in data:
        return jsonify({"error": "Invalid request payload"}), 400
    try:
        prediction_df = pd.DataFrame(data["data"])

        # Load the ensemble model
        ensemble_model = joblib.load("ensemble_model.pkl")

        # Make predictions
        predictions = ensemble_model.predict(prediction_df)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {e}"}), 500

# Download endpoint
@app.route('/download_model', methods=['GET'])
def download_model():
    model_path = "best_model.pkl"
    if os.path.exists(model_path):
        return jsonify({"message": "Model is available", "model_path": model_path})
    else:
        return jsonify({"error": "Model file not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)