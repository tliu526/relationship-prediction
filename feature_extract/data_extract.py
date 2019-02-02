"""Script for raw data extraction.

"""

import argparse
import csv
import math
import multiprocessing
import os
import pickle
import shutil
from sys import exit

import numpy as np
import pandas as pd

from df_utils import *


coe_cols =  ["timestamp", "contact_name", "contact_number", "comm_type", "comm_direction"]


"""
eml data extraction functions
"""
def extract_eml_data(data_dir, subj, testing=False):
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


"""
eml feature construction functions/declarations
"""
canonical_visit_reasons = ['Entertainment',
                           'Errand',
                           'Home',
                           'Work',
                           'Exercise',
                           'Dining',
                           'Socialize',
                           'Travelling / Traffic'
                          ]

canonical_locs = ["Home", 
                  "Work", 
                  "Another's Home", 
                  "Arts & Entertainment (Theater, Music Venue, Etc.)",
                  "Food (Restaurant, Cafe)",
                  'Nightlife Spot (Bar, Club)',
                  'Outdoors & Recreation',
                  'Gym or Other Exercise',
                  'Professional or Medical Office',
                  'Spiritual (Church, Temple, Etc.)',
                  'Shop or Store',
                  'Travel or Transport (Airport, Bus Stop, Train Station, Etc.)',
                  'Vehicle'
                 ]

short_name_locs = ["loc:home",
                   "loc:work",
                   "loc:anothers_home",
                   "loc:arts/entertainment",
                   "loc:food",
                   "loc:nightlife",
                   "loc:outdoors/recreation",
                   "loc:gym/exercise",
                   "loc:professional/medical_office",
                   "loc:spiritual",
                   "loc:shop",
                   "loc:travel/transport",
                   "loc:vehicle",
                   "loc:other"
                ]

canonical_visit_reasons = ['Entertainment',
                           'Errand',
                           'Home',
                           'Work',
                           'Exercise',
                           'Dining',
                           'Socialize',
                           'Travelling / Traffic'
                          ]

short_name_visit_reasons = ['visit_reason:entertainment',
                            'visit_reason:errand',
                            'visit_reason:home',
                            'visit_reason:work',
                            'visit_reason:exercise',
                            'visit_reason:dining',
                            'visit_reason:socialize',
                            'visit_reason:travel/traffic',
                            'visit_reason:other'
                           ]


def map_locations(locations):
    """Takes the locations array as input and expands into a Series.
    
    """
    col_dict = {k:0 for k in short_name_locs}
    
    if type(locations) is float:
        return pd.Series(col_dict)
    
    for loc in locations:
        if loc in canonical_locs:
            col_dict[short_name_locs[canonical_locs.index(loc)]] = 1
        else:
            col_dict['other'] = 1
            
    return pd.Series(col_dict)
    

def map_visit_reasons(visit_reasons):
    """Takes the visit_reasons array as input and expands into a Series.
    
    """
    col_dict = {k:0 for k in short_name_visit_reasons}
    
    if type(visit_reasons) is float:
        return pd.Series(col_dict)
    
    for visit in visit_reasons:
        if visit in canonical_visit_reasons:
            col_dict[short_name_visit_reasons[canonical_visit_reasons.index(visit)]] = 1
        else:
            col_dict['visit_reason:other'] = 1
            
    return pd.Series(col_dict)


def merge_eml(eml_df, coe_df):
    """Processes eml_df into usable columns and merges with coe_df.
    
    """
    merge_cols = ['pid', 'timestamp', 'location', 'visit_reason']

    eml_df['timestamp'] = eml_df['timestamp'].astype(float)
    eml_df = eml_df.drop_duplicates(subset=['pid', 'timestamp'])
    merged_df = coe_df.merge(eml_df[merge_cols], on=['pid', 'timestamp'], how='left')

    both_pids = list(set(eml_df['pid']).intersection(set(coe_df['pid'])))
    merged_df = merged_df.loc[merged_df['pid'].isin(both_pids)]

    # feature column construction
    merged_df['visit_reason'] = merged_df['visit_reason'].map(lambda x: eval(x), na_action='ignore')
    merged_df['location'] = merged_df['location'].map(lambda x: eval(x), na_action='ignore')

    loc_df = merged_df['location'].apply(map_locations)
    visit_df = merged_df['visit_reason'].apply(map_visit_reasons)

    final_df = pd.concat([merged_df, loc_df, visit_df], axis=1)

    return final_df

"""
cal processing and feature construction
"""
def process_cal(cal_df, pid):
    """
    Processes the cal.csv call logs to determine call duration.
    
    """
    cols = ['pid', 'ring_start', 'ring_end', 'ring_duration', 'call_start', 'call_end', 'call_duration']
    
    start_call_ts = 0
    end_call_ts = 0
    start_ring_ts = np.nan
    end_ring_ts = np.nan
    in_call = False
    is_ring = False
    prev_state = None
    call_rows = []
    for idx, row in cal_df.iterrows():
        # starts ringing
        if (not in_call) and row['call_state'] == 'Ringing':
            is_ring = True
            start_ring_ts = row['timestamp']
        
        # call begins
        elif (not in_call) and (row['call_state'] == 'Off-Hook'):
            # pick
            if prev_state == 'Ringing':
                end_ring_ts = row['timestamp']
            start_call_ts = row['timestamp']
            in_call = True
        
        # call ends
        elif in_call and (row['call_state'] == 'Idle'):
            in_call = False
            end_call_ts = row['timestamp']
            if is_ring:
                is_ring = False
            else:
                start_ring_ts = np.nan
                end_ring_ts = np.nan
            call_rows.append([pid, start_ring_ts, end_ring_ts, end_ring_ts - start_ring_ts, start_call_ts, end_call_ts, end_call_ts - start_call_ts])

        # the missed call case
        elif (not in_call) and (row['call_state'] == 'Idle'):
            is_ring = False
            start_ring_ts = np.nan
            end_ring_ts = np.nan
            
        prev_state = row['call_state']
    
    proc_df = pd.DataFrame(call_rows, columns=cols)

    return proc_df



    
def match_dates(phone_df, cal_df):
    """Matches call length from cal_df to the given logs in phone_df.
    
    """
    cols = ['combined_hash', 'timestamp', 'ring_duration', 'call_duration']
    combined_rows = []
    for idx, row in phone_df.iterrows():
        ts = math.ceil(row['timestamp'])
        cur_call = row['comm_direction']
        if cur_call in ['INCOMING', 'OUTGOING']:
            for cal_idx, cal_row in cal_df.iterrows():
                start = cal_row['ring_start'] if not np.isnan(cal_row['ring_start']) else cal_row['call_start']
                start -= 30
                end = cal_row['call_end']+1

                if (start <= ts) and (end >= ts):
                    data = [row['combined_hash'], row['timestamp'], cal_row['ring_duration'], cal_row['call_duration']]
                    combined_rows.append(data)
                    break

    pid_df = pd.DataFrame(combined_rows, columns=cols)
    return pid_df


def extract_cal_data(data_dir, subj, coe_df, testing=False):
    """Extracts call length information for the given subject and merges with coe data
    
    """
    filename = data_dir + subj + '/cal.csv'
    call_cols = ['timestamp', 'call_state']

    if os.path.exists(filename):
        print(filename)

        with open(filename) as file_in:
            raw_df = pd.read_csv(file_in, delimiter='\t', header=None, names=call_cols)
            cal_df = process_cal(raw_df, subj)
            pid_coe = coe_df.loc[coe_df['pid'] == subj]
            phone_df = pid_coe.loc[pid_coe['comm_type'] == 'PHONE']
            
            return match_dates(phone_df, cal_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='directory with all participant data')
    parser.add_argument('coe_file', help='raw aggregated coe df') # TODO move coe processing here too
    parser.add_argument('out_name', help='output file name')
    parser.add_argument('num_processes', type=int, help='number of processes to spin up')
    parser.add_argument('--test', action='store_true', help='whether to make a test run of the data extraction')
    parser.add_argument('--eml', action='store_true', help='whether to process the semantic location eml.csvs')
    parser.add_argument('--cal', action='store_true', help='whether to process the call duration cal.csvs')

    args = parser.parse_args()
    subjects = os.listdir(args.data_dir)
    
    # TODO move coe processing to this file
    with open(args.coe_file, 'rb') as coe_f:
        coe_df = pickle.load(coe_f)

    final_df = coe_df.copy()

    if args.eml:
        eml_args = [(args.data_dir, subj, args.test) for subj in subjects]
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            results = pool.starmap(extract_eml_data, eml_args)

        eml_data = pd.DataFrame()
        for df in results:
            eml_data = eml_data.append(df)

        final_df = merge_eml(eml_data, coe_df)

    if args.cal:
        if args.test:
            subjects = subjects[:1]
        cal_args = [(args.data_dir, subj, final_df, args.test) for subj in subjects]  
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            results = pool.starmap(extract_cal_data, cal_args)

        cal_df = pd.DataFrame()
        for df in results:
            cal_df = cal_df.append(df)
            

        final_df = final_df.merge(cal_df, on=['combined_hash', 'timestamp'], how='outer') # TODO verify this is the correct behavior

    final_df = final_df.drop_duplicates(subset=['timestamp', 'pid', 'combined_hash'])
    pickle.dump(final_df, open("{}.df".format(args.out_name), 'wb'), -1)

    