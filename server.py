import os
import numpy as np
import torch
import json
from flask import Flask, request, jsonify
from pathlib import Path

from compile import get_model, get_dataset, run_fhe_model
from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer

app = Flask(__name__)

# Global variables to store model and keys
model_name = "phenoage"
fhe_directory = str(Path("fhe_models") / Path(model_name))
client = None
server = None
model_cls = None

def initialize_fhe_services():
    global client, server, model_cls
    
    # Get dataset for model initialization (if needed)
    dataset_np = get_dataset(model_name)
    
    # Get the model
    cml_model, model_cls = get_model(model_name, dataset_np)
    
    # Create FHE directory if it doesn't exist
    if not os.path.exists(fhe_directory):
        dev = FHEModelDev(
            path_dir=fhe_directory,
            model=cml_model)
        dev.save()
    
    # Setup the client
    client = FHEModelClient(path_dir=fhe_directory, key_dir="/tmp/keys_client")
    
    # Setup the server
    server = FHEModelServer(path_dir=fhe_directory)
    server.load()
    
    print("FHE services initialized successfully")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.json
        
        if not data or 'features' not in data:
            return jsonify({"error": "Invalid input: 'features' field is required"}), 400
        
        # Convert input to numpy array
        input_data = np.array(data['features'], dtype=np.float64)
        
        if len(input_data.shape) == 1:
            # Convert single sample to 2D array
            input_data = input_data.reshape(1, -1)
            
        # Preprocess data
        input_data = model_cls.preprocess(torch.tensor(input_data)).numpy()
        
        # Generate encryption keys
        serialized_evaluation_keys = client.get_serialized_evaluation_keys()
        
        # Encrypt the data
        encrypted_data = client.quantize_encrypt_serialize(input_data)
        
        # Server processes the encrypted data
        encrypted_result = server.run(encrypted_data, serialized_evaluation_keys)
        
        # Client decrypts the result
        result = client.deserialize_decrypt_dequantize(encrypted_result)
        
        # Postprocess the result
        final_result = model_cls.postprocess(torch.tensor(result))
        
        # Convert tensor to Python list for JSON serialization
        final_result_list = final_result.tolist()
        
        return jsonify({
            "predictions": final_result_list
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Initialize FHE models and services
    initialize_fhe_services()
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)