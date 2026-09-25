# No-Shrink Hierarchical Pooling

## Version requirements:

* Python 3.10.18
* Tensorflow 2.16.2
* Spektral 1.3.1
* Numpy 1.26.4
* Optuna 4.6.0


## Data

CONTCAR_files.zip contains all crystal structure data used in this work. The folder "data_partitions" contains several csv files which detail exact training, validation, and test partitions, cation and anion groupings according to heuristics, and target values for each crystal structure.

The heuristic used for cation and anion groupings uses covalent bond radii and electronegativity values to determine which atoms are most likely to be covalently bonded. Code to generate these heuristic poolings is in heuristic_functions.py.

## Code

zintl_optuna_sweep.py contains the code for our hyperparameter sweep experiments, and clean_prediction_script.py contains the code for evaluating experiments, including generating files needed for pooling evaluation.

spektral_essential_objects.py contains all customized objects and architectures required for our experiments, including AtomFeaDataset, NoShrinkDiffPool, and NotShrinking architecture. utils.py contains several functions used for training or evaluation, including pooling evaluation.


## Results

charge_analysis_ELF.zip contains ELFCAR files, which are the result of electron localization function analysis on some crystal structures that were used in this work. This analysis was used to manually evaluate the quality of our poolings.

NoShrink_results_and_saved_model.zip contains our trained NoShrink model, associated hyperparameters, and prediction and pooling results for all datasets used in this work.

cgcnn_results.zip contains the trained CGCNN model used for comparisons, associated hyperparameters, prediction results for all datasets used in this work.

## Citation
This work is available on ChemRxiv as a preprint and has not yet been peer-reviewed. Citation information will be updated after peer-review.

Nila Mandal, Rinkumoni Chaliha, Prashun Gorai, Qian Yang. Unsupervised Motif Discovery in Zintl Phases via Sparse Hierarchical Graph Learning. ChemRxiv. 13 August 2026.
DOI: https://doi.org/10.26434/chemrxiv.15007407/v1
