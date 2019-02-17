#!/bin/bash

# AutoML runs for tie strength score prediction

python run_automl.py ../data/final_features/all_tie_str_baseline final_results/tie_str/tie_str_baseline_automl_reg tie_str_score --run_time 1440 --task_time 21600 > tie_str_baseline_automl_reg.out;

python run_automl.py ../data/final_features/all_tie_str_all final_results/tie_str/tie_str_all_automl_reg tie_str_score --run_time 1440 --task_time 21600  > tie_str_all_automl_reg.out;


