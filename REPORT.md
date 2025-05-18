# Implement an FHE-based Biological Age and Aging Pace Estimation ML Model Using Zama Libraries

Implementation of epigentic age prediction using Concrete-ML.

## List of Models

- [PhenoAge](https://pmc.ncbi.nlm.nih.gov/articles/PMC5940111/)
- [DunedinPACE](https://elifesciences.org/articles/73420)

## Model Justification

The above models are compared based on their required biological samples and hazard ratios.

## Process

### 1. Port Models

Firstly we port the models from PyTorch to Concrete-ML.
Most of them are ElasticNet models so we can just take the weights and biases
and overwrite the Concrete-ML ElasticNet model.
We then save these models, which allows them to be swapped within the
HuggingFace space.

### 2. Benchmark Models

## Results

### PhenoAge

MAE (FHE ElasticNet vs `pyaging`): 0.004873293246623689 years