import gradio as gr
import pandas as pd
import numpy as np
from io import StringIO

from lib import run_fhe_model

PHENOAGE = """,albumin,creatinine,glucose,log_crp,lymphocyte_percent,mean_cell_volume,red_cell_distribution_width,alkaline_phosphatase,white_blood_cell_count,age
patient1,51.8,87.2,4.5,-0.2,27.9,92.4,13.9,123.5,6.037100000000001,70.2"""

# Sample data for testing
def get_sample_data(model_name):
    samples = {
        "PhenoAge (Levine)": PHENOAGE,
    }
    csv_data = samples.get(model_name, "")
    # Convert CSV string to DataFrame
    if csv_data:
        return pd.read_csv(StringIO(csv_data))
    return pd.DataFrame()

# Process uploaded file
def process_file(file_path):
    if file_path is None:
        return pd.DataFrame()
    
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return pd.DataFrame()

# Simple predict function for testing
def predict(data_source, sample_data, uploaded_file, model_name):
    # Determine which data to use based on data_source
    if data_source == "upload" and uploaded_file is not None:
        df = process_file(uploaded_file)
    else:
        df = sample_data
        
    print(f"Received data for model {model_name}, shape: {df.shape}")
    
    # Use df.empty to check if DataFrame is empty instead of "if not df"
    if df.empty:
        return {"Status": "Error: No data provided"}
    
    try:
        # Simulate a prediction result based on the model
        if model_name == "PhenoAge (Levine)":
            biological_age = np.random.normal(45, 10)
            aging_pace = np.random.normal(1.0, 0.2)
        
        return {
            "Biological Age": round(biological_age, 1),
            "Aging Pace": round(aging_pace, 2),
            "Model Used": model_name,
            "Status": f"Processed with {model_name} using FHE"
        }
            
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return {"Status": f"Error: {str(e)}"}

# Create a simple test app
def create_test_app():
    models = [
        "PhenoAge (Levine)",
    ]
    
    with gr.Blocks() as demo:
        gr.Markdown("""# Biological Age Estimation

[Zama AI Bounty](https://github.com/zama-ai/bounty-program/issues/143?utm_campaign=49915104-Bounty%20Program&utm_medium=email&_hsmi=109371217&utm_content=109371217&utm_source=hs_email)""")
        
        with gr.Row():
            with gr.Column(scale=1):
                model_selector = gr.Dropdown(
                    choices=models,
                    value=models[0],  # PhenoAge is default
                    label="Select Model"
                )
                
                data_source = gr.Radio(
                    choices=["sample", "upload"],
                    value="sample",
                    label="Data Source"
                )
            
            with gr.Column(scale=2):
                # Use DataFrame component instead of Textbox
                sample_data = gr.DataFrame(
                    value=get_sample_data("PhenoAge (Levine)"),
                    label="Sample Data",
                    interactive=True,
                    visible=True
                )
                
                file_upload = gr.File(
                    label="Upload CSV File",
                    file_types=[".csv"],
                    visible=False
                )
                
                predict_btn = gr.Button("Predict Biological Age")
                output = gr.JSON(label="Results")
        
        # Update sample data when model changes
        model_selector.change(
            fn=get_sample_data,
            inputs=model_selector,
            outputs=sample_data
        )
        
        # Toggle visibility based on data source selection
        def toggle_visibility(choice):
            return gr.update(visible=choice == "sample"), gr.update(visible=choice == "upload")
        
        data_source.change(
            fn=toggle_visibility,
            inputs=data_source,
            outputs=[sample_data, file_upload]
        )
        
        # Prediction handler
        predict_btn.click(
            fn=predict,
            inputs=[data_source, sample_data, file_upload, model_selector],
            outputs=output
        )
    
    return demo

if __name__ == "__main__":
    print("Starting local test app...")
    app = create_test_app()
    app.launch()