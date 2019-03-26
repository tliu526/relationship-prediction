#!/bin/bash

# AutoML runs for subpopulation prediction

# AutoML baseline features

## Q1
python run_automl.py ../data/subpop_features/top5_base_q1 subpop_results/q1_base contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res > subpop_logs/q1_base.out
## Q2
python run_automl.py ../data/subpop_features/top5_base_q2 subpop_results/q2_base contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res > subpop_logs/q2_base.out
## Q3
python run_automl.py ../data/subpop_features/top5_base_q3 subpop_results/q3_base contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res > subpop_logs/q3_base.out
## Q4
python run_automl.py ../data/subpop_features/top5_base_q4 subpop_results/q4_base contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res > subpop_logs/q4_base.out


# AutoML all features

## Q1
python run_automl.py ../data/subpop_features/top5_all_q1 subpop_results/q1_all contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res > subpop_logs/q1_all.out
## Q2
python run_automl.py ../data/subpop_features/top5_all_q2 subpop_results/q2_all contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res > subpop_logs/q2_all.out
## Q3
python run_automl.py ../data/subpop_features/top5_all_q3 subpop_results/q3_all contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res > subpop_logs/q3_all.out
## Q4
python run_automl.py ../data/subpop_features/top5_all_q4 subpop_results/q4_all contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res > subpop_logs/q4_all.out


# random forest baseline features

## Q1
python run_automl.py ../data/subpop_features/top5_base_q1 subpop_results/q1_base_rf contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res --rand_forest > subpop_logs/q1_base_rf.out
## Q2
python run_automl.py ../data/subpop_features/top5_base_q2 subpop_results/q2_base_rf contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res --rand_forest > subpop_logs/q2_base_rf.out
## Q3
python run_automl.py ../data/subpop_features/top5_base_q3 subpop_results/q3_base_rf contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res --rand_forest > subpop_logs/q3_base_rf.out
## Q4
python run_automl.py ../data/subpop_features/top5_base_q4 subpop_results/q4_base_rf contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res --rand_forest > subpop_logs/q4_base_rf.out


# random forest all features

## Q1
python run_automl.py ../data/subpop_features/top5_all_q1 subpop_results/q1_all_rf contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res --rand_forest > subpop_logs/q1_all_rf.out
## Q2
python run_automl.py ../data/subpop_features/top5_all_q2 subpop_results/q2_all_rf contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res --rand_forest > subpop_logs/q2_all_rf.out
## Q3
python run_automl.py ../data/subpop_features/top5_all_q3 subpop_results/q3_all_rf contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res --rand_forest > subpop_logs/q3_all_rf.out
## Q4
python run_automl.py ../data/subpop_features/top5_all_q4 subpop_results/q4_all_rf contact_type --weighted_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res --rand_forest > subpop_logs/q4_all_rf.out
