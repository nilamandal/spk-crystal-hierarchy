# spk-crystal-hierarchy

Version requirements:

* Python 3.10.18
* Tensorflow 2.16.2
* Spektral 1.3.1
* Numpy 1.26.4
* Optuna 4.6.0


Files:

* ExampleContcars is a directory containing examples of the data formats used in this work.
* spektral_essential_objects.py contains all customized objects and architectures required for our experiments, including AtomFeaDataset, NoShrinkDiffPool, and NotShrinking architecture.
* zintl_optuna_sweep.py uses Optuna to perform a large hyperparameter sweep.
* clean_prediction_script.py generates all plots and spreadsheets typically needed to evaluate a trained model.
* utils.py contains several functions used for training or evaluation, including pooling evaluation.
* cgcnn_pca.py contains code used to analyze CGCNN features with PCA and MDS.  
* generate_plots.py generates multiple types of plots, mainly plots of predicted value vs target value for different models.
* clustering_comparison.py uses sklearn's implementation of DBSCAN to cluster atoms. It also computes several clustering performance metrics.
* NoShrink_results_and_saved_model.zip contains our trained NoShrink model, associated hyperparameters, and prediction and pooling results for all datasets used in this work.
* cgcnn_results.zip contains the trained CGCNN model used for comparisons, associated hyperparameters, prediction results for all datasets used in this work, and results from PCA and MDS analysis of learned features.
