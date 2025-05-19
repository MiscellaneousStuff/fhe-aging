import gradio as gr
import pandas as pd
import numpy as np
import requests
import json
import threading
import time
import subprocess
import os
import signal
from io import StringIO

# Sample data for testing
PHENOAGE = """,albumin,creatinine,glucose,log_crp,lymphocyte_percent,mean_cell_volume,red_cell_distribution_width,alkaline_phosphatase,white_blood_cell_count,age
patient1,51.8,87.2,4.5,-0.2,27.9,92.4,13.9,123.5,6.037100000000001,70.2"""

# Global variables for server management
server_process = None
server_ready = False

def start_server():
    global server_process, server_ready
    
    print("Starting FHE model server...")
    server_ready = False
    # Start the server as a subprocess
    server_process = subprocess.Popen(["python", "server.py"])
    
    # Wait for server to become available
    attempts = 0
    max_attempts = 30  # Maximum attempts to connect to server
    
    while attempts < max_attempts:
        try:
            response = requests.get("http://localhost:5000/health")
            if response.status_code == 200:
                print("Server is ready!")
                server_ready = True
                break
        except requests.exceptions.ConnectionError:
            pass
        
        time.sleep(1)
        attempts += 1
    
    if not server_ready:
        print("Failed to start server after multiple attempts")

def stop_server():
    global server_process, server_ready
    
    if server_process:
        print("Stopping server...")
        # Send termination signal to server process
        if os.name == 'nt':  # Windows
            server_process.terminate()
        else:  # Unix/Linux/MacOS
            os.kill(server_process.pid, signal.SIGTERM)
        
        server_process.wait()
        server_ready = False
        print("Server stopped")

# Get sample data based on model name
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

# Predict function that communicates with the Flask server
def predict(data_source, sample_data, uploaded_file, model_name):
    global server_ready
    
    # Map Gradio model names to server model names
    model_map = {
        "PhenoAge (Levine)": "phenoage"
    }
    
    server_model = model_map.get(model_name, "phenoage")
    
    # Determine which data to use based on data_source
    if data_source == "upload" and uploaded_file is not None:
        df = process_file(uploaded_file)
    else:
        df = sample_data
    
    print(f"Received data for model {model_name}, shape: {df.shape if df is not None else 'None'}")
    
    # Check if DataFrame is empty
    if df.empty:
        return {"Status": "Error: No data provided"}
    
    try:
        # Check if server is running
        if not server_ready:
            return {"Status": "Error: Server not ready. Please try again in a moment."}
        
        # Extract just the feature columns (all except age if it exists)
        feature_cols = df.columns.tolist()
        
        # Use only the numeric columns as features
        features = df[feature_cols].select_dtypes(include=[np.number]).values.tolist()
        
        if not features:
            return {"Status": "Error: No numeric features found in data"}
        
        # Prepare payload for API request
        payload = {
            "model": server_model,
            "features": features
        }
        
        # Make API call to local server
        response = requests.post("http://localhost:5000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            # return {"Status": "Success", "Result": result["predictions"][0]}
            return result["predictions"][0]
        else:
            error_msg = response.json().get("error", "Unknown server error")
            # return {"Status": f"Server Error: {error_msg}"}
            return error_msg
    
    except requests.exceptions.ConnectionError:
        return {"Status": "Error: Cannot connect to server"}
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return {"Status": f"Error: {str(e)}"}

# Create a simple test app
def create_test_app():
    models = [
        "PhenoAge (Levine)",
    ]
    
    with gr.Blocks() as demo:
        gr.Markdown("""# Biological Age Estimation with FHE
[Zama AI Bounty](https://github.com/zama-ai/bounty-program/issues/143?utm_campaign=49915104-Bounty%20Program&utm_medium=email&_hsmi=109371217&utm_content=109371217&utm_source=hs_email)

This application uses Fully Homomorphic Encryption (FHE) to make predictions while keeping your data encrypted.
""")
        
        with gr.Row():
            with gr.Column(scale=1):
                # server_status = gr.Textbox(
                #     label="Server Status",
                #     value="Not started",
                #     interactive=False
                # )
                
                # start_server_btn = gr.Button("Start Server")
                # stop_server_btn = gr.Button("Stop Server")
                
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
                # Use DataFrame component
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
        
        # # Server management functions
        # def update_server_status(action):
        #     global server_ready
            
        #     if action == "start":
        #         if server_ready:
        #             return "Server is already running"
        #         else:
        #             # Start server in a separate thread
        #             thread = threading.Thread(target=start_server)
        #             thread.daemon = True
        #             thread.start()
        #             return "Starting server... Please wait"
            
        #     elif action == "stop":
        #         if not server_ready:
        #             return "Server is not running"
        #         else:
        #             stop_server()
        #             return "Server stopped"
            
        #     return "Unknown action"
        
        # start_server_btn.click(
        #     fn=lambda: update_server_status("start"),
        #     inputs=[],
        #     outputs=server_status
        # )
        
        # stop_server_btn.click(
        #     fn=lambda: update_server_status("stop"),
        #     inputs=[],
        #     outputs=server_status
        # )
        
        # # Auto-check server status periodically
        # def check_server_status():
        #     global server_ready
            
        #     try:
        #         response = requests.get("http://localhost:5000/health")
        #         if response.status_code == 200:
        #             server_ready = True
        #             return "Server is running"
        #         else:
        #             server_ready = False
        #             return "Server is not responding properly"
        #     except:
        #         server_ready = False
        #         return "Server is not running"
        
        # demo.load(
        #     fn=check_server_status,
        #     inputs=[],
        #     outputs=[],
        # )
        
        # Prediction handler
        predict_btn.click(
            fn=predict,
            inputs=[data_source, sample_data, file_upload, model_selector],
            outputs=output
        )
    
    return demo

if __name__ == "__main__":
    print("Starting application...")
    
    # Start the server in the background
    thread = threading.Thread(target=start_server)
    thread.daemon = True
    thread.start()
    
    # Start Gradio app
    app = create_test_app()
    
    # Clean up server when exiting
    try:
        app.launch()
    finally:
        stop_server()