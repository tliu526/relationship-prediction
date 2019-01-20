"""Module for Pandas dataframe processing utilities."""

import csv
import math
import os

import numpy as np
import pandas as pd

__all__ = ['get_time_from_gps', 'get_location_from_gps', 'preprocess_location']


def get_time_from_gps(path, time_now, time_prev, lat_report, lng_report):
    """Finds the start and ending timestamps given a particular location.

    Taken from Sohrab Saeb's CS120DataAnalysis repository: https://github.com/sosata/CS120DataAnalysis/blob/master/SemanticLocation/get_time_from_gps.py

    """

    filename = path + '/fus.csv'
    t = []
    lat = []
    lng = []
    if os.path.isfile(filename):
        with open(filename) as file_in:
            data = csv.reader(file_in, delimiter='\t')
            for data_row in data:
                if data_row:
                    t.append(float(data_row[0]))
                    lat.append(float(data_row[1]))
                    lng.append(float(data_row[2]))
        file_in.close()
    else:
        print('error: location file not found')
        return np.array([]), np.array([])

    # limiting data to current window
#    dif = [abs(x-time_prev) for x in t]
#    ind_start = dif.index(min(dif))
#    dif = [abs(x-time_now) for x in t]
#    ind_end = dif.index(min(dif))
#    t = t[ind_start:ind_end]
#    lat = lat[ind_start:ind_end]
#    lng = lng[ind_start:ind_end]
    lat = [lat[i] for i in range(len(t)) if (t[i]>time_prev and t[i]<time_now)]
    lng = [lng[i] for i in range(len(t)) if (t[i]>time_prev and t[i]<time_now)]
    t = [t[i] for i in range(len(t)) if (t[i]>time_prev and t[i]<time_now)]

    # filtering gps data based on location
    d = [math.sqrt((lat[i]-lat_report)**2+(lng[i]-lng_report)**2) for i in range(len(lat))]
    ind_within = [i for i in range(len(d)) if d[i]<.001]
    t = [t[i] for i in ind_within]

    if not t:
        print('no data - instance skipped')
        return np.array([]), np.array([])

    # finding isolated t's
    inds = [i for i in range(len(t)-1) if t[i+1]-t[i]>600]
    t_end = []
    t_start = [t[0]]
    for i in range(len(inds)):
        t_end.append(t[inds[i]])
        t_start.append(t[inds[i]+1])
    t_end.append(t[len(t)-1])
    
    if not t:
        print('no data - instance skipped')
        return np.array([]), np.array([])
        
    return t_start, t_end


def get_data_at_location(path, t_start, t_end, sensor_name):
    """Extracts the given sensor data from the specified timeframe.

    Utilizes t_start, t_end from get_time_from_gps.
    Taken from Sohrab Saeb's CS120DataAnalysis repository: https://github.com/sosata/CS120DataAnalysis/blob/master/SemanticLocation/get_data_at_location.py

    """

    filename = path + '/' + sensor_name + '.csv'
    if os.path.isfile(filename):
        with open(filename) as file_in:
            data = csv.reader(file_in, delimiter='\t')
            data_value = []
            for data_row in data:
                if data_row:
                    for i in range(len(t_start)):
                        if (float(data_row[0])>=t_start[i])and(float(data_row[0])<=t_end[i]):
                            #print 'data added'
                            data_value.append(data_row)
        file_in.close()
    else:
        print(('warning: sensor '+sensor_name+' not found.'))
        return []

    return data_value


def remove_extra_characters(x):
    x = x.replace('"','')
    x = x.replace('[','')
    x = x.replace(']','')
    return x

def remove_extra_space(x):
    if type(x)==str:
        if x.startswith(' '):
            x = x[1:]
        if x.endswith(' '):
            x = x[:-1]
        y = x
    else:
        y = []
        for xi in x:
            if xi.startswith(' '):
                xi = xi[1:]
            if xi.endswith(' '):
                xi = xi[:-1]
            if xi:
                y += [xi]
    return y

def title_case(x):
    y = x.title()
    # make letters after apostrophe lowercase
    if "'" in y:
        ind = y.find("'")
        y_temp = ''
        for (j,l) in enumerate(y):
            if j==ind+1:
                y_temp += l.lower()
            else:
                y_temp += l
        y = y_temp
    return y
        
def remove_parentheses(ss):
    s = ss.split('(')
    ss = s[0]
    if ss.endswith(' '):
        ss = ss[:-1]
    return ss

def correct_order(x):
    if ',' in x:
        x_parsed = x.split(',')
        x_parsed = [remove_extra_space(x_p) for x_p in x_parsed]
        x_parsed = sorted(x_parsed)
        x = ','.join(x_parsed)
    return x

def preprocess_reason(reason, parse=True):
    
    # if only one element is sent
    if type(reason)==str:
        reason = remove_extra_characters(reason)
        reason = remove_extra_space(reason)
        if '\\u2026' in reason:
            reason = reason.replace('\\u2026','')
        reason = reason.lower()
        reason = correct_order(reason)
    else:
        reason_new = []
        for r in reason:
            r = remove_extra_characters(r)
            if '\\u2026' in r:
                r = r.replace('\\u2026','')
            if r:
                # parsing
                if ',' in r and parse:
                    r_parsed = r.split(',')
                    if '' in r_parsed:
                        r_parsed = filter(None, r_parsed) # removing empty strings
                    if r_parsed:
                        r_parsed = [r_p.lower() for r_p in r_parsed]
                else:
                    r_parsed = [r.lower()]
                
                # removing extra start or end spaces
                r_parsed = remove_extra_space(r_parsed)

                if r_parsed:
                    reason_new += r_parsed
        reason = reason_new
    
    return reason


def preprocess_location(location, parse=True):
    """Processes the given location string.

    Taken from Sohrab Saeb's CS120DataAnalysis repository: https://github.com/sosata/CS120DataAnalysis/blob/master/SemanticLocation/preprocess.py

    """

    # if only one element is sent
    if type(location)==str:
        location = remove_extra_characters(location)
        location = remove_extra_space(location)
        location = title_case(location)
        location = remove_parentheses(location)
    else:
        location_new = []
        for l in location:
            l = remove_extra_characters(l)
            if l:
                # parsing
                if ',' in l and parse:
                    l_parsed = l.split(',')
                    if '' in l_parsed:
                        l_parsed = filter(None, l_parsed) # removing empty strings
                    if l_parsed:
                        l_parsed = [l_p for l_p in l_parsed]
                        #l_parsed = [l_p.upper() for l_p in l_parsed]
                else:
                    l_parsed = [l]
                    #l_parsed = [l.upper()]

                # removing extra start or end spaces
                l_parsed = remove_extra_space(l_parsed)

                if l_parsed:
                    location_new += l_parsed
        location = location_new

    return location
