#!/bin/bash
# runs autoML with original baseline features

python run_automl.py top_5_emc results/top_5_contact_type_baseline contact_type 
python run_automl.py top_5_emc results/top_5_q1_want_baseline q1_want
python run_automl.py top_5_emc results/top_5_q2_talk_baseline q2_talk 
python run_automl.py top_5_emc results/top_5_q3_loan_baseline q3_loan 
python run_automl.py top_5_emc results/top_5_q4_closeness_baseline q4_closeness 

