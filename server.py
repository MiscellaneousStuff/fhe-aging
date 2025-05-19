import os
import numpy as np
import torch
import json
from flask import Flask, request, jsonify
from pathlib import Path

from compile import get_model, get_dataset, run_fhe_model
from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer

app = Flask(__name__)

# Dictionary to store model-specific resources
models = {}

def initialize_model(model_name):
    """Initialize a specific model and its FHE components"""
    if model_name in models:
        # Model already initialized
        return True
    
    try:
        fhe_directory = str(Path("fhe_models") / Path(model_name))
        
        # Get dataset for model initialization
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
        client = FHEModelClient(path_dir=fhe_directory, key_dir=f"/tmp/keys_client_{model_name}")
        
        # Setup the server
        server = FHEModelServer(path_dir=fhe_directory)
        server.load()
        
        # Store all model-specific components
        models[model_name] = {
            "client": client,
            "server": server,
            "model_cls": model_cls
        }
        
        print(f"Model '{model_name}' initialized successfully")
        return True
    
    except Exception as e:
        print(f"Error initializing model '{model_name}': {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

@app.route('/available_models', methods=['GET'])
def available_models():
    """Return a list of available models"""
    # Currently only 'phenoage' is supported, but this can be extended
    return jsonify({"available_models": ["phenoage"]})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.json
        
        if not data:
            return jsonify({"error": "Invalid request: no JSON data provided"}), 400
        
        # Get model name from request, default to "phenoage" if not specified
        model_name = data.get('model', 'phenoage')
        
        if not data.get('features'):
            return jsonify({"error": "Invalid input: 'features' field is required"}), 400
        
        # Initialize the model if not already done
        if model_name not in models:
            success = initialize_model(model_name)
            if not success:
                return jsonify({"error": f"Failed to initialize model '{model_name}'"}), 500
        
        # Get model components
        model_components = models[model_name]
        client = model_components["client"]
        server = model_components["server"]
        model_cls = model_components["model_cls"]
        
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
            "model": model_name,
            "predictions": final_result_list
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Initialize the default model (phenoage)
    initialize_model("phenoage")
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)