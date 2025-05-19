import pandas as pd 
import numpy as np

from sklearn.linear_model import ElasticNet as SKLearnElasticNet
from concrete.ml.sklearn import ElasticNet as CMElasticNet
from anndata.experimental.pytorch import AnnLoader

from pyaging.models import *
from pyaging.predict._postprocessing import *
from pyaging.predict._preprocessing import *

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
    dataset_np = get_dataset("phenoage")

    # Get model
    cml_model, model_cls = get_model("phenoage", dataset_np)

    # Inference
    result = run_fhe_model(cml_model, dataset_np, model_cls)
    print(result)