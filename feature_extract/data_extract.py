"""Script for raw data extraction.

"""

import argparse
import csv
import multiprocessing
import os
import pickle
import shutil
from sys import exit

import numpy as np
import pandas as pd

from df_utils import *


coe_cols =  ["timestamp", "contact_name", "contact_number", "comm_type", "comm_direction"]


def extract_loc_coe_data(data_dir, subj, testing=False):
    """Extracts semantic location data for given communications.
    
    """
    loc_coe_df = pd.DataFrame()
    filename = data_dir + subj + '/eml.csv'
    if os.path.exists(filename):
        print(filename)
        loc = []
        lat_report = []
        lng_report = []
        t_report = []
        with open(filename) as file_in:
            data = csv.reader(file_in, delimiter='\t')
            eml = []
            for data_row in data:
                if data_row:
                    # reading location category (state)
                    loc_string = data_row[6]
                    loc_string = loc_string[1:len(loc_string)-1]
                    loc_string.split(',')
                    loc.append(loc_string)
                    
                    # reading lat. and long.
                    lat_report.append(float(data_row[2]))
                    lng_report.append(float(data_row[3]))
                    t_report.append(float(data_row[0]))
                    
                    # adding to eml
                    eml.append(data_row)
                    
        file_in.close()
    else:
        print('skipping subject '+subj+' without location report/foursquare data.')
        return
        
                      
    # looking into data between current and previous report
    filename = data_dir + subj + '/fus.csv'
    if os.path.exists(filename):
        with open(filename) as file_in:
            data_gps = csv.reader(file_in, delimiter='\t')
            t_gps = []
            lat_gps = []
            lng_gps = []
            for row_gps in data_gps:
                if row_gps:
                    t_gps.append(float(row_gps[0]))
                    lat_gps.append(float(row_gps[1]))
                    lng_gps.append(float(row_gps[2]))
        file_in.close()
    else:
        print('skipping subject '+subj+' without location data.')
        return
    
    t_prev = 0


    for (i,eml_row) in enumerate(eml):

        # finding t_start and t_end from gps data
        t_start, t_end = get_time_from_gps(data_dir+subj, t_report[i], t_prev, lat_report[i], lng_report[i])

        # if there is any clusters found, extract sensor data and put in a separate file
        if len(t_start)>0:
            data = get_data_at_location(data_dir+subj, t_start, t_end, 'coe')
            if len(data)>0:
                df = pd.DataFrame(data, columns=coe_cols)
                df['pid'] = subj
                df['location'] = eml_row[6] # location label(s)
                df['visit_reason'] = eml_row[7] # semantic location visit reason

                loc_coe_df = loc_coe_df.append(df)
                if testing:
                    return loc_coe_df
        else:
            print('instance '+str(i)+' skipped')
            
        # continue iteration
        if i<len(t_report)-1:
            if t_report[i]!=t_report[i+1]:
                t_prev = t_report[i]
                
    return loc_coe_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='directory with all participant data')
    parser.add_argument('out_name', help='output file name')
    parser.add_argument('num_processes', type=int, help='number of processes to spin up')
    parser.add_argument('--test', action='store_true', help='whether to make a test run of the model training')

    args = parser.parse_args()
    
    subjects = os.listdir(args.data_dir)

    func_args = [(args.data_dir, subj, args.test) for subj in subjects]

    with multiprocessing.Pool(processes=args.num_processes) as pool:
        results = pool.starmap(extract_loc_coe_data, func_args)

    final_df = pd.DataFrame()
    for df in results:
        final_df = final_df.append(df)
    
    pickle.dump(final_df, open("{}.df".format(args.out_name), 'wb'), -1)

    