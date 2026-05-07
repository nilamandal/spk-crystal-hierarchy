# spk-crystal-hierarchy

bohb_experiment_setup.py contains full hyperparameter tuning loop using bayes opt hyperband.

clean_prediction_script.py generates all plots and spreadsheets typically needed to evaluate a trained model.

scaffold_split.py contains the code used to replicate the split from the other benchmarked models experiments, split code is held in bench_test, train, val, accordingly

aqsol_experiments.py starts job, runs training loop in utils.py
eval.py collects all finished jobs and results into a csv
