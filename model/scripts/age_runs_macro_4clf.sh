#!/bin/bash

# AutoML runs for out of sample test set evaluation of age features, optimized for macro F1

# AutoML features

## communication features baseline
python run_automl.py ../data/age_features/top5_comm age_results/comm_automl_macro contact_type --macro_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res > age_logs/comm_automl_macro.out

## age features baseline
python run_automl.py ../data/age_features/top5_age age_results/age_automl_macro contact_type --macro_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res > age_logs/age_automl_macro.out

## age + comm features
python run_automl.py ../data/age_features/top5_age_comm age_results/age_comm_automl_macro contact_type --macro_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res > age_logs/age_comm_automl_macro.out

## all features
python run_automl.py ../data/age_features/top5_all age_results/all_automl_macro contact_type --macro_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res > age_logs/all_automl_macro.out


# random forest features

# communication features baseline
python run_automl.py ../data/age_features/top5_comm age_results/comm_rf_macro contact_type --macro_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res --rand_forest > age_logs/comm_rf_macro.out

## age features baseline
python run_automl.py ../data/age_features/top5_age age_results/age_rf_macro contact_type --macro_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res --rand_forest > age_logs/age_rf_macro.out

## age + comm features
python run_automl.py ../data/age_features/top5_age_comm age_results/age_comm_rf_macro contact_type --macro_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res --rand_forest > age_logs/age_comm_rf_macro.out

## all features
python run_automl.py ../data/age_features/top5_all age_results/all_rf_macro contact_type --macro_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res --rand_forest > age_logs/all_rf_macro.out



