import gradio as gr
import pandas as pd
import numpy as np
from io import StringIO

from lib import run_model_fhe

# Sample data for testing
def get_sample_data(model_name):
    samples = {
        # "DNA Methylation (Horvath)": "cg00000292,cg00002426,cg00003994\n0.782,0.496,0.901",
        "PhenoAge (Levine)": "albumin,creatinine,glucose,c_reactive_protein\n4.2,0.9,85,1.2",
        # "DunedinPACE": "cg05575921,cg21566642,cg01940273\n0.892,0.734,0.211",
        # "GrimAge": "feature1,feature2,feature3,feature4\n0.5,0.6,0.7,0.8"
    }
    return samples.get(model_name, "")

# Process uploaded file
def process_file(file_path):
    if file_path is None:
        return ""
    
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""

# Simple predict function for testing
def predict(data_source, sample_data, uploaded_file, model_name):
    # Determine which data to use based on data_source
    if data_source == "upload" and uploaded_file is not None:
        data = process_file(uploaded_file)
    else:
        data = sample_data
        
    print(f"Received data for model {model_name}: {data[:50]}...")
    
    if not data:
        return {"Status": "Error: No data provided"}
    
    try:
        # Convert string data to dataframe
        df = pd.read_csv(StringIO(data))
        print(f"Parsed data shape: {df.shape}")
        
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
        # "DNA Methylation (Horvath)",
        "PhenoAge (Levine)",
        # "DunedinPACE",
        # "GrimAge"
    ]
    
    with gr.Blocks() as demo:
        gr.Markdown("""# Biological Age Estimation - LOCAL TEST

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
                sample_data = gr.Textbox(
                    label="Sample Data (CSV format)", 
                    value=get_sample_data("PhenoAge (Levine)"),
                    lines=10,
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