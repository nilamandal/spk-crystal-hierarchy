# No-Shrink Hierarchical Pooling

## Version requirements:

* Python 3.10.18
* Tensorflow 2.16.2
* Spektral 1.3.1
* Numpy 1.26.4
* Optuna 4.6.0


## Data

## Code

## Results

## Citation
This work is available on ChemRxiv as a preprint and has not yet been peer-reviewed. Citation information will be updated after peer-review.

Nila Mandal, Rinkumoni Chaliha, Prashun Gorai, et al. Unsupervised Motif Discovery in Zintl Phases via Sparse Hierarchical Graph Learning. ChemRxiv. 13 August 2026.
DOI: https://doi.org/10.26434/chemrxiv.15007407/v1

Files:

* heuristic_functions.py includes code to identify cation and anionic motifs using a heuristic based on covalent bond radii and electronegativity.
* spektral_essential_objects.py contains all customized objects and architectures required for our experiments, including AtomFeaDataset, NoShrinkDiffPool, and NotShrinking architecture.
* zintl_optuna_sweep.py uses Optuna to perform a large hyperparameter sweep.
* clean_prediction_script.py generates all plots and spreadsheets typically needed to evaluate a trained model.
* utils.py contains several functions used for training or evaluation, including pooling evaluation.
* charge_analysis_ELF.zip contains ELFCAR files, which are the result of electron localization function analysis on some crystal structures that were used in this work. This analysis was used to manually evaluate the quality of our poolings.
* generate_plots.py generates multiple types of plots, mainly plots of predicted value vs target value for different models.
* clustering_comparison.py uses sklearn's implementation of DBSCAN to cluster atoms. It also computes several clustering performance metrics.
* NoShrink_results_and_saved_model.zip contains our trained NoShrink model, associated hyperparameters, and prediction and pooling results for all datasets used in this work.
* cgcnn_results.zip contains the trained CGCNN model used for comparisons, associated hyperparameters, prediction results for all datasets used in this work.
