import os
import pandas as pd 
import numpy as np
from pathlib import Path

from sklearn.linear_model import ElasticNet as SKLearnElasticNet

from pyaging.models import *
from pyaging.predict._postprocessing import *
from pyaging.predict._preprocessing import *

from concrete.ml.sklearn import ElasticNet as CMElasticNet
from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer

class PhenoAge(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        """
        Applies a convertion from a CDF of the mortality score from a Gompertz
        distribution to phenotypic age.
        """
        # lambda
        l = torch.tensor(0.0192, device=x.device, dtype=x.dtype)
        mortality_score = 1 - torch.exp(-torch.exp(x) * (torch.exp(120 * l) - 1) / l)
        age = 141.50225 + torch.log(-0.00553 * torch.log(1 - mortality_score)) / 0.090165
        return age

def get_dataset(dataset_name: str):
    # Init Dataset
    df = pd.read_pickle('pyaging_data/blood_chemistry_example.pkl')

    # Dataset
    dataset = df.iloc[:, 0:10]
    dataset_np = dataset.to_numpy()
    return dataset_np

def get_model(model_name: str, dataset_np: np.array):
    assert model_name in ["phenoage"]
    if model_name == "phenoage":
        print(">>> get model: sklearn elasticnet")
        # Model to SKLearn
        manual_coefficients = np.array([
            -0.0336,  0.0095,  0.1953,  0.0954, -0.0120,  0.0268,  0.3306,  0.0019, 0.0554,  0.0804
        ])
        manual_intercept = np.array([
            -19.9067
        ])
        sklearn_model = SKLearnElasticNet()
        sklearn_model.n_features_in_ = len(manual_coefficients)
        sklearn_model.coef_ = manual_coefficients
        sklearn_model.intercept_ = manual_intercept
        phenoage_cls = PhenoAge()

        print(">>> get model: cm elasticnet")
        print(">>> get model: cm elasticnet -> init model")
        cml_model = CMElasticNet.from_sklearn_model(
            sklearn_model, dataset_np, n_bits=16)
        print(">>> get model: cm elasticnet -> compile model")
        cml_model.compile(dataset_np)
        return cml_model, phenoage_cls

def run_fhe_model(
    cm_model: CMElasticNet,
    X: np.array,
    handler_cls: pyagingModel
) -> torch.tensor:
    print(">>> run fhe model")
    preprocessed_X = handler_cls.preprocess(X)
    y_pred_fhe = cm_model.predict(preprocessed_X, fhe="execute")
    result = handler_cls.postprocess(torch.tensor(y_pred_fhe))
    return result

if __name__ == "__main__":
    model = "phenoage"

    fhe_directory = str(Path("fhe_models") / Path(model))
    
    dataset_np = get_dataset(model)
    cml_model, model_cls = get_model(model, dataset_np)

    if not os.path.exists(fhe_directory):
        dev = FHEModelDev(
            path_dir=fhe_directory,
            model=cml_model)
        dev.save()

    # Input Data
    input_data = dataset_np[0:1, :]
    input_data = model_cls.preprocess(torch.tensor(input_data)).numpy()

    # Setup the client
    client = FHEModelClient(path_dir=fhe_directory, key_dir="/tmp/keys_client")
    serialized_evaluation_keys = client.get_serialized_evaluation_keys()
    encrypted_data = client.quantize_encrypt_serialize(input_data)

    # Setup the server
    server = FHEModelServer(path_dir=fhe_directory)
    server.load()

    # Server processes the encrypted data
    encrypted_result = server.run(encrypted_data, serialized_evaluation_keys)

    # Client decrypts the result
    result = client.deserialize_decrypt_dequantize(encrypted_result)
    final_result = model_cls.postprocess(torch.tensor(result))
    print(final_result)