#!/bin/bash

# AutoML runs for out of sample test set evaluation of age features

# AutoML features

## communication features baseline
python run_automl.py ../data/age_features/top5_comm age_results/comm_automl contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res > age_logs/comm_automl.out

## age features baseline
python run_automl.py ../data/age_features/top5_age age_results/age_automl contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res > age_logs/age_automl.out

## age + comm features
python run_automl.py ../data/age_features/top5_age_comm age_results/age_comm_automl contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res > age_logs/age_comm_automl.out

## all features
python run_automl.py ../data/age_features/top5_all age_results/all_automl contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res > age_logs/all_automl.out


# random forest features

# communication features baseline
python run_automl.py ../data/age_features/top5_comm age_results/comm_rf contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res --rand_forest > age_logs/comm_rf.out

## age features baseline
python run_automl.py ../data/age_features/top5_age age_results/age_rf contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res --rand_forest > age_logs/age_rf.out

## age + comm features
python run_automl.py ../data/age_features/top5_age_comm age_results/age_comm_rf contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res --rand_forest > age_logs/age_comm_rf.out

## all features
python run_automl.py ../data/age_features/top5_all age_results/all_rf contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res --rand_forest > age_logs/all_rf.out



