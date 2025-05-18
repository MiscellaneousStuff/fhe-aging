#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[1]:


import pandas as pd 
import pyaging as pya


# In[2]:


pya.data.download_example_data('blood_chemistry_example')


# In[3]:


df = pd.read_pickle('pyaging_data/blood_chemistry_example.pkl')


# In[4]:


adata = pya.preprocess.df_to_adata(df)


# In[5]:


import marshal
import math
import ntpath
import os
import types
from typing import Dict, List, Tuple
from urllib.request import urlretrieve

import anndata
import numpy as np
import pandas as pd
import torch
from anndata.experimental.pytorch import AnnLoader
from torch.utils.data import DataLoader, TensorDataset

from pyaging.logger import LoggerManager, main_tqdm, silence_logger
from pyaging.models import *
from pyaging.utils import download, load_clock_metadata, progress
from pyaging.predict._postprocessing import *
from pyaging.predict._preprocessing import *

@progress("Predict ages with model")
def predict_ages_with_model(
    adata: anndata.AnnData,
    model: pyagingModel,
    device: str,
    batch_size: int,
    logger,
    indent_level: int = 2,
) -> torch.Tensor:
    """
    Predict biological ages using a trained model and input data.

    This function takes a machine learning model and input data, and returns predictions made by the model.
    It's primarily used for estimating biological ages based on various biological markers. The function
    assumes that the model is already trained. A dataloader is used because of possible memory constraints
    for large datasets.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the dataset. Its `.X` attribute is expected to be a matrix where rows
        correspond to samples and columns correspond to features.

    model : pyagingModel
        The pyagingModel of the aging clock of interest.

    device : str
        Device to move AnnData to during inference. Eithe 'cpu' or 'cuda'.

    batch_size : int
        Batch size for the AnnLoader object to predict age.

    logger : Logger
        A logger object for logging the progress or any relevant information during the prediction process.

    indent_level : int, optional
        The indentation level for logging messages, by default 2.

    Returns
    -------
    predictions : torch.Tensor
        An array of predicted ages or biological markers, as returned by the model.

    Notes
    -----
    Ensure that the data is preprocessed (e.g., scaled, normalized) as required by the model before
    passing it to this function. The model should be in evaluation mode if it's a type that has different
    behavior during training and inference (e.g., PyTorch models).

    The exact nature of the predictions (e.g., age, biological markers) depends on the model being used.

    Examples
    --------
    >>> model = load_pretrained_model()
    >>> predictions = predict_ages_with_model(model, "cpu", logger)
    >>> print(predictions[:5])
    [34.5, 29.3, 47.8, 50.1, 42.6]

    """

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


# In[6]:


pya.pred.predict_age(adata, predict_ages_with_model, 'PhenoAge')


# In[7]:


adata.obs.head()


# In[8]:


adata.obs["phenoage"]

