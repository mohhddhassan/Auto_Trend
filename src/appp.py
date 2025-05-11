import streamlit as st
import pandas as pd
import requests
import os
from pycaret.regression import *
from pycaret.regression import pull, load_model
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Streamlit UI 
st.set_page_config(page_title="AutoML", layout="wide")

# Loading dataset
if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv')

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload", "Data Preprocessing", "Profiling", "Modelling", "Ensemble", "Prediction", "Download"])
    st.info("This project application helps you build and explore your data.")

# Uploading part 
if choice == "Upload":
    st.title("Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload Your Dataset (CSV format)", type=["csv"])
    if uploaded_file:
        try:
            # Display the uploaded file in Streamlit
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)

            # Send file to Flask backend for processing
            files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
            response = requests.post("http://localhost:5000/upload", files=files)

            # Check the Flask response
            if response.status_code == 200:
                st.success(response.json().get("message", "File uploaded successfully!"))
                df.to_csv('dataset.csv', index=False)  # Save locally for profiling
            else:
                st.error(f"Error uploading file: {response.text}")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

# Data Preprocessing 
if choice == "Data Preprocessing":
    st.title("Data Preprocessing")
    if 'df' in locals():
        st.write("Here are the first few rows of the dataset:")
        st.dataframe(df.head())

        # Handle missing values
        st.subheader("Handle Missing Values")
        missing_value_strategy = st.selectbox(
            "Select a strategy for missing values", ["Drop Rows", "Impute with Mean/Median"]
        )
        if missing_value_strategy == "Drop Rows":
            df = df.dropna()
            st.success("Rows with missing values have been dropped.")
        elif missing_value_strategy == "Impute with Mean/Median":
            imputer = SimpleImputer(strategy='mean')  # You can also use 'median'
            df[df.columns] = imputer.fit_transform(df)
            st.success("Missing values have been imputed with the mean/median.")

        # Categorical feature encoding
        st.subheader("Encode Categorical Variables")
        encode_option = st.selectbox("Select encoding method", ["None", "Label Encoding", "One-Hot Encoding"])
        if encode_option == "Label Encoding":
            le = LabelEncoder()
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = le.fit_transform(df[col])
            st.success("Categorical variables have been label encoded.")
        elif encode_option == "One-Hot Encoding":
            df = pd.get_dummies(df)
            st.success("Categorical variables have been one-hot encoded.")

        # Feature scaling
        st.subheader("Feature Scaling")
        scale_option = st.selectbox("Select scaling method", ["None", "Standard Scaling", "Min-Max Scaling"])
        if scale_option == "Standard Scaling":
            scaler = StandardScaler()
            df[df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))
            st.success("Features have been standardized (z-score normalization).")
        elif scale_option == "Min-Max Scaling":
            df[df.select_dtypes(include=['float64', 'int64']).columns] = (df.select_dtypes(include=['float64', 'int64']) - df.min()) / (df.max() - df.min())
            st.success("Features have been scaled using Min-Max scaling.")

        # Save processed data
        df.to_csv('dataset.csv', index=False)
        st.write("Processed dataset:")
        st.dataframe(df.head())

    else:
        st.warning("Please upload a dataset first.")

# Profiling the dataset
if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    if 'df' in locals():
        from ydata_profiling import ProfileReport
        from streamlit_pandas_profiling import st_profile_report

        profile_df = ProfileReport(df, explorative=True)
        st_profile_report(profile_df)
    else:
        st.warning("Please upload a dataset first.")

# Modelling
if choice == "Modelling":
    st.title("Model Training")
    if 'df' in locals():
        target_column = st.selectbox("Choose the Target Column", df.columns)
        if st.button("Train Model"):
            try:
                # Validate the target column
                if df[target_column].isnull().any():
                    st.error(f"Target column '{target_column}' contains missing values. Please handle missing values in the 'Data Preprocessing' section.")
                elif not pd.api.types.is_numeric_dtype(df[target_column]):
                    st.error(f"Target column '{target_column}' must be numeric. Please encode or transform the column in the 'Data Preprocessing' section.")
                else:
                    # Prepare JSON data for Flask model training
                    data_payload = {
                        "data": df.to_dict(orient="records"),
                        "target": target_column
                    }
                    response = requests.post("http://localhost:5000/model", json=data_payload)

                    # Display response
                    if response.status_code == 200:
                        model_details = response.json()
                        st.success(model_details.get("message", "Model trained successfully!"))

                        # Display top 5 models in a table
                        st.subheader("Top 5 Models")
                        top_5_models = model_details.get("top_5_models", [])
                        if top_5_models:
                            # Create a DataFrame for the table
                            table_data = {
                                "Rank": [model["rank"] for model in top_5_models],
                                "Model Name": [model["model_name"] for model in top_5_models],
                                "RMSE": [model["rmse"] for model in top_5_models],
                                "MAE": [model["mae"] for model in top_5_models],
                                "R²": [model["r2"] for model in top_5_models]
                            }
                            df_table = pd.DataFrame(table_data)
                            st.table(df_table)
                        else:
                            st.warning("No models were returned.")
                    else:
                        st.error(f"Error during model training: {response.text}")
            except Exception as e:
                st.error(f"An error occurred during model training: {e}")
    else:
        st.warning("Please upload and profile your dataset first.")
# Ensemble Models
if choice == "Ensemble":
    st.title("Ensemble Models")
    if 'df' in locals():
        if st.button("Create Ensemble Model"):
            try:
                response = requests.post("http://localhost:5000/ensemble")
                if response.status_code == 200:
                    ensemble_details = response.json()
                    st.success(ensemble_details.get("message", "Ensemble model created successfully!"))
                    st.json(ensemble_details.get("ensemble_details", {}))
                else:
                    st.error(f"Error during ensemble creation: {response.text}")
            except Exception as e:
                st.error(f"An error occurred during ensemble creation: {e}")
    else:
        st.warning("Please upload and profile your dataset first.")

# Prediction
if choice == "Prediction":
    st.title("Make Predictions")
    if 'df' in locals():
        st.write("Upload a CSV file for prediction:")
        prediction_file = st.file_uploader("Upload Prediction Dataset (CSV format)", type=["csv"])
        if prediction_file:
            try:
                prediction_df = pd.read_csv(prediction_file)
                st.dataframe(prediction_df.head())

                # Send prediction data to Flask backend
                prediction_payload = {
                    "data": prediction_df.to_dict(orient="records")
                }
                response = requests.post("http://localhost:5000/predict", json=prediction_payload)

                # Display predictions
                if response.status_code == 200:
                    predictions = response.json().get("predictions", [])
                    st.success("Predictions generated successfully!")
                    st.write(predictions)
                else:
                    st.error(f"Error during prediction: {response.text}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please upload and profile your dataset first.")

# Download the trained model
if choice == "Download":
    st.title("Download Trained Model")
    try:
        response = requests.get("http://localhost:5000/download_model")
        if response.status_code == 200:
            model_path = response.json().get("model_path", "best_model.pkl")
            with open(model_path, "rb") as file:
                st.download_button("Download Trained Model", file, file_name="best_model.pkl")
        else:
            st.error(response.json().get("error", "Model file not found."))
    except Exception as e:
        st.error(f"An error occurred: {e}")