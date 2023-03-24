# spk-crystal-hierarchy

bayestuner directory contains incomplete code to apply a bayesian hyperparameter tuner to our architectures.

parallel_process_workflow.py coordinates data loading, the training process, the hyperparameter tuning process, and plots the learning curves.

scaling_structures.py scales unrelaxed structures.

spektral_essential_objects.py contains all relevant objects for data handling, architecture classes, etc.

plotlearningcurves.py plots hyperparameter-vs-performance scatterplots.

spektral_pred.py runs predictions on trained saved models and prints assignment matrices to a text file.

assignmentplots.py plots individual crystal's pool assignments.

testing.py organizes training output txt files into spreadsheets to better organize the hyperparameter tuning process.
