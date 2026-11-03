# deb-jepa

Development fork of Diogo Oliveira's DEB initialization implementation. This
checkout adds a bounded supervised readiness runner; JEPA training is future work.
The upstream README, attribution and citation instructions follow below.

- [Training readiness and verified commands](docs/training-readiness.md)
- [Baseline evidence](docs/baseline-evidence.json)
- [Compact readiness result](docs/readiness-summary.json)
- Research roadmap and teaching materials live in the sibling Tese repository.
  Local entry point: `../Tese/README.md`; the repositories must be checked out as siblings.

---

# DEB ML Init

## Reliable machine learning initialization methods for the calibration of Dynamic Energy Budget models

To obtain a DEB model, the calibration problem must be solved: find the parameters that minimize the deviation
between observed data and model predictions. While DEB model calibration is largely automated, the selection of initial 
parameters remains a key unresolved step, since the only automated method – the bijection method – often fails to 
produce a feasible initial parameter set. Consequently, modelers resort to trial-and-error to find parameters to seed
the estimation. To bridge this gap, we propose using machine learning to initialize the calibration. We develop two
models: a neural network and a 1-nearest-neighbor. Both models are built with a focus on feasibility, directly
integrating parameter constraints into their structure. 

This repository contains the code and notebooks used in the paper: 


This repository contains the code and notebooks used to build datasets from the Add‑my‑Pet (AmP) collection, 
train ML models that predict DEB parameters from commonly available zero‑variate data plus taxonomy/ecology, 
check feasibility, and compare against the bijection baseline. 
The repository also includes the DEBtool implementation of the algorithms and code to convert Python models to MATLAB models compatible with DEBtool.

---


## Repository Structure

```
DEB_ML_Init/
├─ data/ — the raw data derived from AmP and processed datasets. See README file inside.
├─ notebooks/ — Jupyter notebooks to build datasets, train/evaluate models, and reproduce results from the paper
├─ project_code/ — Python and MATLAB code for data processing, model training and inference, feasibility checks, and calibration hand‑off
│  ├─ algorithms/ — Implements the neural network, the taxonomy 1‑NN, and the bijection method
│  ├─ data/ — dataset building and preprocessing scripts
│  ├─ debtool_port/ — Implementation in MATLAB of the ML initialization methods for DEBtool integration
│  ├─ evaluate/ — metrics and model evaluation scripts for feasibility and initialization performance
│  ├─ inference/ — scripts to run inference on trained models
│  ├─ calibration/ — MATLAB/DEBtool glue + Python helpers to launch calibrations
│  ├─ plotters/ — plotting utilities for results visualization
│  ├─ train/ — scripts to train and tune hyperparameters of the ML models
│  └─ utils/ — general utility functions
├─ results/ — saved models, metrics, and plots from experiment runs
├─ filepaths.csv — central config of local paths (edit to reflect your setup)
└─ requirements.txt — Python dependency list
```

---

## Usage

### Python

#### Neural Networks

The neural network models are implemented in PyTorch at:
- `project_code/algorithms/neural_networks.py`: contains the model definitions
- `project_code/algorithms/loss_functions.py`: contains the custom loss functions used to train the models
- `project_code/data/prepare_data_pytorch.py`: data loading and preprocessing scripts for PyTorch models

The scripts to train the model are in `project_code/train/`:
- `train_pytorch_model.py`: training script
- `calibrate_pytorch_model.py`: script to tune hyperparameters 



#### 1-Nearest-Neighbor
The taxonomy 1-NN models are implemented using scikit-learn at `project_code/algorithms/`:
- `project_code/algorithms/taxonomic_1nn.py`: contains the model definition
- `project_code/algorithms/build_sklearn_model.py`: contains helper functions to build and save the model
- `project_code/data/prepare_data_sklearn.py`: data loading and preprocessing scripts for scikit-learn models

### DEBtool

Make sure to download the latest available DEBtool version at https://github.com/add-my-pet/DEBtool_M/.
To run the InitNet initialization, the code in the directory `DEBtool M/lib/pet/DEBInitNet` must be added to the 
MATLAB path alongside the other DEBtool directories. Its usage is similar to other initialization methods in DEBtool. 
Simply choose option ’3’ as the pars init method:

```
estim_options(’pars_init_method’, 3);
```
---


## Reproducing the paper

The notebooks reproduce the experiments and figures from the manuscript:

- **6.1. Feasibility analysis**: `notebooks/Evaluate ML models.ipynb`
- **6.2. Prediction performance of proposed machine learning models**: `notebooks/Compare models on feasibility.ipynb` 
- **6.3. Initialization performance**: `notebooks/Compare models on initialization performance.ipynb`

---

## How to cite

If you use this code or trained models, please cite:

> Oliveira D.F., Marques G.M., Santos F.M.P., Pecquerie L., Sousa J.M.C., Domingos T. (2025). Reliable machine learning initialization methods for the calibration of Dynamic Energy Budget models. SSRN. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5434102

If you use the bijection method as a baseline, please also cite:

> Lika K., Augustine S., Pecquerie L., Kooijman S.A.L.M. (2014). The bijection from data to parameter space with the standard DEB model quantifies the supply–demand spectrum. Journal of Theoretical Biology, 354:35–47. https://doi.org/10.1016/j.jtbi.2014.03.025

---

## Acknowledgements

We thank the Add‑my‑Pet community and DEBtool maintainers for making DEB modeling broadly accessible and for help with 
the integration of DEBMLInit methods in DEBtool.
