#!/bin/bash

# AutoML runs for tie strength rank prediction, replicates svm

python run_automl.py ../data/final_features/all_tie_str_baseline final_results/tie_str/tie_str_baseline_svm tie_str_class --run_time 1440 --task_time 21600 --svc > tie_str_baseline_svm.out;

python run_automl.py ../data/final_features/all_tie_str_all final_results/tie_str/tie_str_all_svm tie_str_class --run_time 1440 --task_time 21600 --svc > tie_str_all_svm.out;


