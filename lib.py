import pandas as pd 
import pyaging as pya
import numpy as np
import torch
import anndata

from sklearn.linear_model import ElasticNet as SKLearnElasticNet
from concrete.ml.sklearn import ElasticNet as CMElasticNet
from anndata.experimental.pytorch import AnnLoader

from pyaging.models import *
from pyaging.predict._postprocessing import *
from pyaging.predict._preprocessing import *
from pyaging.utils import progress
from pyaging.logger import main_tqdm

@progress("Predict ages with model")
def predict_ages_with_model(
    adata: anndata.AnnData,
    model: pyagingModel,
    device: str,
    batch_size: int,
    logger,
    indent_level: int = 2,
) -> torch.Tensor:
    # If there is a preprocessing step
    if model.preprocess_name is not None:
        logger.info(
            f"The preprocessing method is {model.preprocess_name}",
            indent_level=indent_level + 1,
        )
    else:
        logger.info("There is no preprocessing necessary", indent_level=indent_level + 1)

    # If there is a postprocessing step
    if model.postprocess_name is not None:
        logger.info(
            f"The postprocessing method is {model.postprocess_name}",
            indent_level=indent_level + 1,
        )
    else:
        logger.info("There is no postprocessing necessary", indent_level=indent_level + 1)

    # Create an AnnLoader
    use_cuda = torch.cuda.is_available()
    dataloader = AnnLoader(adata, batch_size=batch_size, use_cuda=use_cuda)

    # with torch.no_grad():
    #     for param in model.parameters():
    #         param.zero_()

    # Use the AnnLoader for batched prediction
    predictions = []
    with torch.inference_mode():
        for batch in main_tqdm(dataloader, indent_level=indent_level + 1, logger=logger):
            batch_pred = model(batch.obsm[f"X_{model.metadata['clock_name']}"])
            predictions.append(batch_pred)
    # Concatenate all batch predictions
    predictions = torch.cat(predictions)

    return predictions

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

def run_fhe_model(
    cm_model: CMElasticNet,
    X: np.array,
    handler_cls: pyagingModel
) -> torch.tensor:
    preprocessed_X = handler_cls.preprocess(X)
    y_pred_fhe = cm_model.predict(preprocessed_X, fhe="execute")
    result = handler_cls.postprocess(torch.tensor(y_pred_fhe))
    return result

if __name__ == "__main__":
    # Init Dataset
    df = pd.read_pickle('pyaging_data/blood_chemistry_example.pkl')
    adata = pya.preprocess.df_to_adata(df)
    pya.pred.predict_age_fhe(adata, predict_ages_with_model, 'PhenoAge')

    # Load Model Weights
    device = "cpu"
    weights_path = f"./pyaging_data/phenoage.pt"
    clock = torch.load(weights_path, weights_only=False)
    for name, param in clock.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
    clock.to(torch.float64)
    clock.to(device)
    clock.eval()

    # Dataset
    dataset = df.iloc[:, 0:10]
    dataset_np = dataset.to_numpy()
    phenoages_np = np.array(adata.obs["phenoage"], dtype=np.float64)
    dataset_torch = torch.tensor(dataset_np, dtype=torch.float64)
    phenoages_torch = torch.tensor(phenoages_np, dtype=torch.float64)
    with torch.inference_mode():
        pred = clock(dataset_torch)

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

    # FHE
    phenoage_cls = PhenoAge()
    cml_model = CMElasticNet.from_sklearn_model(
        sklearn_model, dataset_np, n_bits=16)
    cml_model.compile(dataset_np)

    # Inference
    result = run_fhe_model(cml_model, dataset_np, phenoage_cls)
    print(result)