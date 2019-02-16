#!/bin/bash

# AutoML runs for replicating Min et al 3-class life facet classification with SVMs

python run_automl.py ../data/zimmerman_features/zimmerman_contact_type_baseline final_results/zimmerman/zimmerman_contact_type_baseline_svm contact_type --resample --rand_downsample --zimmerman_classes --run_time 1440 --task_time 21600 --svc > zimmerman_baseline_svm.out;

python run_automl.py ../data/zimmerman_features/zimmerman_contact_type_all final_results/zimmerman/zimmerman_contact_type_all_svm contact_type --resample --rand_downsample --zimmerman_classes --run_time 1440 --task_time 21600 --svc  > zimmerman_all_svm.out;
