#!/bin/bash

# AutoML runs for subpopulation prediction, 4clf and micro f1 as target

# AutoML all features

## Q1
python run_automl.py ../data/subpop_features/top5_all_q1 subpop_results/q1_all_micro contact_type --micro_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res > subpop_logs/q1_all_micro.out
## Q2
python run_automl.py ../data/subpop_features/top5_all_q2 subpop_results/q2_all_micro contact_type --micro_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res > subpop_logs/q2_all_micro.out
## Q3
python run_automl.py ../data/subpop_features/top5_all_q3 subpop_results/q3_all_micro contact_type --micro_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res > subpop_logs/q3_all_micro.out
## Q4
python run_automl.py ../data/subpop_features/top5_all_q4 subpop_results/q4_all_micro contact_type --micro_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res > subpop_logs/q4_all_micro.out
## seeded runs
python run_automl.py ../data/subpop_features/top5_all_allq_sX subpop_results/allq_sX_all_micro contact_type --micro_f1 --collapse_classes --run_time 1440 --task_time 21600 --group_res > subpop_logs/allq_sX_all_micro.out

