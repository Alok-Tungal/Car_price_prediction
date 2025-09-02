import gradio as gr
import pandas as pd
from skops.hub_utils import download
import joblib

# --- Configuration ---
# The repo_id of the model you pushed to the Hub
REPO_ID = "your-hf-username/cars24-price-predictor" # <-- IMPORTANT: REPLACE
LOCAL_MODEL_PATH = "hf_model_repository/model.joblib"

# --- Load Model from Hugging Face Hub ---
# Download the model files from the hub to a local directory
download(repo_id=REPO_ID, dst="hf_model_repository")
# Load the model pipeline from the local file
model_pipeline = joblib.load(LOCAL_MODEL_PATH)
print("Model loaded successfully!")

# --- Prediction Function ---
def predict_price(year, km_driven, fuel_type, transmission):
    """
    Takes car features as input and returns the predicted price.
    """
    # Create a pandas DataFrame from the inputs
    input_data = pd.DataFrame({
        'year': [year],
        'km_driven': [km_driven],
        'fuel_type': [fuel_type],
        'transmission': [transmission]
    })
    
    # Make a prediction
    prediction = model_pipeline.predict(input_data)
    
    # Format the output
    predicted_price = round(prediction[0], 2)
    return f"Predicted Price: {predicted_price} Lakhs"

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚗 Cars24 Used Car Price Predictor")
    gr.Markdown("Enter the details of the car to get an estimated selling price.")
    
    with gr.Row():
        year = gr.Slider(minimum=2010, maximum=2025, step=1, label="Manufacturing Year", value=2018, info="Select the year the car was made")
        km_driven = gr.Slider(minimum=1000, maximum=200000, step=1000, label="Kilometers Driven", value=50000, info="Enter the total kilometers driven")
    
    with gr.Row():
        fuel_type = gr.Radio(choices=['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'], label="Fuel Type", value="Petrol")
        transmission = gr.Radio(choices=['Manual', 'Automatic'], label="Transmission", value="Manual")

    predict_btn = gr.Button(value="Predict Price", variant="primary")
    output_price = gr.Label(label="Result")
    
    predict_btn.click(
        fn=predict_price,
        inputs=[year, km_driven, fuel_type, transmission],
        outputs=output_price
    )

# Launch the app
demo.launch()
