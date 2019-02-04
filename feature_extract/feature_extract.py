"""Module for extracting communication features.

All functions take as input a Dataframe with the following columns:
- comm_direction: ["INCOMING", "OUTGOING"]
- comm_type: ["PHONE", "SMS"]
- pid: str
- date: DateTime
- timestamp: int (epoch time)
- day: int
- hour: int
- hour_wk: int

TODO turn DataFrame into a class for better contract?
"""

import numpy as np
import pandas as pd
from pandas.tseries.holiday import Holiday, AbstractHolidayCalendar, USThanksgivingDay

DAY_DIVISOR = 6  # 6 hour chunks

""" Commenting out for unit tests
__all__ = [
    'comm_feature_extract', 
    'build_nan_features', 
    'build_emc_features',
    'build_demo_features',
    'build_location_features'
]
"""

def comm_feature_extract(comm_df, ema_df):
    """Generates a DataFrame with extracted comm features, one contact per row.

    returns: the DataFrame comm_features
    """

    # prepare dataframes for processing
    comm_df['date_days'] = pd.DatetimeIndex(comm_df['date_tz']).normalize()
    comm_df['time_of_day'] = comm_df['hour'] // DAY_DIVISOR
    call_df = comm_df.loc[comm_df['comm_type'] == 'PHONE']
    sms_df = comm_df.loc[comm_df['comm_type'] == 'SMS']

    comm_features = init_feature_df(comm_df)
    comm_features = build_count_features(comm_features, call_df, sms_df, ema_df)
    comm_features = build_temporal_features(comm_features, call_df, sms_df)
    comm_features = build_intensity_features(comm_features, call_df, sms_df)
    comm_features = build_channel_selection_features(comm_features, comm_df)
    comm_features = build_avoidance_features(comm_features, call_df, sms_df)
    comm_features = build_duration_features(comm_features, call_df)
    comm_features = build_maintenance_features(comm_features, call_df, sms_df)
    comm_features = build_holiday_features(comm_features, comm_df)

    return comm_features


def init_feature_df(raw_df):
    """Initializes the processed feature dataframe from raw_df.

    Should be the first feature transformation method called.

    Features created:
    - total_comms
    - total_comm_days
    - contact_type
    """
    comm_group = raw_df.groupby(['pid', 'combined_hash'])

    # build total counts and comm_features dataframe
    tot_counts = comm_group['contact_type'].count()
    comm_features = tot_counts.to_frame()
    comm_features['total_comms'] = comm_features['contact_type']
    comm_features = comm_features[['total_comms']]

    # build total comm days
    comm_features['total_comm_days'] = comm_group['date_days'].nunique()

    # insert contact_type feature
    single_contacts = raw_df.drop_duplicates('combined_hash')
    comm_features = comm_features.merge(single_contacts[['pid', 'combined_hash', 'contact_type']], 
                                      on=['pid', 'combined_hash'], 
                                      how='outer')

    comm_features = comm_features.set_index(['pid', 'combined_hash'])
    
    return comm_features


def build_count_features(comm_features, call_df, sms_df, ema_df):
    """Returns comm_features with count features built from sms, call dfs.

    ema_df supplies the total number of days each participant is in the study.
    Regularity features are derived from counts.

    Features created:
    - total_calls
    - total_sms
    - total_call_days
    - total_sms_days
    - reg_call: regularity of calls, total_call_days / total_days
    - reg_sms: regularity of sms, total_sms_days / total_days
    - reg_comm: regularity of communication, total_comm_days / total_days
    - total_days
    - total_wks
    """
    call_group = call_df.groupby(['pid', 'combined_hash'])
    sms_group = sms_df.groupby(['pid', 'combined_hash'])

    sms_counts = sms_group['contact_type'].count()
    call_counts = call_group['contact_type'].count()
    comm_features['total_calls'] = call_counts
    comm_features['total_calls'] = comm_features['total_calls'].fillna(0)
    comm_features['total_sms'] = sms_counts
    comm_features['total_sms'] = comm_features['total_sms'].fillna(0)

    comm_features['total_sms_days'] = \
        sms_df.groupby(['pid', 'combined_hash'])['date_days'].nunique()
    comm_features['total_sms_days'] = comm_features['total_sms_days'].fillna(0)
    comm_features['total_call_days'] = \
        call_df.groupby(['pid', 'combined_hash'])['date_days'].nunique()
    comm_features['total_call_days'] = comm_features['total_call_days'].fillna(0)
    comm_features = comm_features.reset_index()

    # build total logged days/weeks for each participant
    total_recorded_days = ema_df.groupby('pid')['date'].nunique()
    comm_features['total_days'] = comm_features.apply(lambda x: total_recorded_days[x.pid], 
                                                      axis=1)

    total_recorded_wks = ema_df.groupby(['pid', pd.Grouper(key='date', freq='W')]).count()
    total_recorded_wks = total_recorded_wks.reset_index()
    total_recorded_wks = total_recorded_wks.groupby('pid')['date'].nunique()
    comm_features['total_wks'] = comm_features.apply(lambda x: total_recorded_wks[x.pid], 
                                                      axis=1)

    comm_features['reg_call'] = comm_features['total_call_days'] / comm_features['total_days']
    comm_features['reg_sms'] = comm_features['total_sms_days'] / comm_features['total_days']
    comm_features['reg_comm'] = comm_features['total_comm_days'] / comm_features['total_days']

    return comm_features


def intensity_helper(comm_features, group_df, col, name):
    """Helper function for build_intensity_features: mean, std, min, med, max

    Feature names will become {mean, std, min, med, max}_{name}

    TODO modify days to weeks
    """

    # for week groupings 
    group_key = ['pid', 'combined_hash', pd.Grouper(key='date_days', freq='W')]

    # drop the other columns 
    group_df = group_df[['pid', 'combined_hash', 'date_days', col]]

    # mean calculation
    mean_name = 'mean_' + name
    comm_sum = group_df.groupby(['pid', 'combined_hash'], as_index=False)[col].sum()
    temp_df = comm_features.merge(comm_sum, on=['pid', 'combined_hash'], how='outer')
    temp_df[mean_name] = temp_df[col] / temp_df['total_wks']

    mean_d = pd.Series(temp_df[mean_name].values,index=temp_df['combined_hash']).to_dict()
    wks_d = pd.Series(temp_df['total_wks'].values,index=temp_df['pid']).to_dict()

    # std calculation
    std_name = 'std_' + name

    # calculate squared sum difference of logged communications 
    group_df['mean_in'] = group_df['combined_hash'].map(mean_d)
    group_df['total_wks'] = group_df['pid'].map(wks_d)
    group_df['ssum'] = (group_df[col] - group_df['mean_in'])**2

    # calculate ssum over delta wks (wks in the study w/o communication)
    wk_group = group_df.groupby(['pid', 'combined_hash'], as_index=False)
    # wk_group = group_df.groupby(['pid', 'combined_hash'], as_index=False)
    ssum_count = wk_group['ssum'].count().copy()
    ssum_count = ssum_count.rename({'ssum': 'ssum_count'}, axis='columns')
    ssum_count['delta_wks'] = ssum_count['pid'].map(wks_d) - ssum_count['ssum_count']
    ssum_count['delta_ssum'] = ((ssum_count['combined_hash'].map(mean_d))**2) * ssum_count['delta_wks']
    
    # add ssum and delta_ssum and calculate std
    ssum_count['ssum'] = wk_group['ssum'].sum()['ssum']
    ssum_count['total_ssum'] = ssum_count['delta_ssum'] + ssum_count['ssum']
    temp_df = temp_df.merge(ssum_count[['pid', 'combined_hash', 'total_ssum']], on=['pid', 'combined_hash'], how='outer')
    temp_df[std_name] = np.sqrt(temp_df['total_ssum'] / (temp_df['total_wks'] - 1))

    # min, med, max over a week
    min_name = 'min_' + name
    med_name = 'med_' + name
    max_name = 'max_' + name
    

    wk_counts = group_df.groupby(group_key)[col].sum()
    
    wk_med = wk_counts.groupby(level=[0,1]).median().reset_index()
    wk_med = wk_med.rename({col: med_name}, axis='columns')
    temp_df = temp_df.merge(wk_med, on=['pid', 'combined_hash'], how='outer')

    wk_min = wk_counts.groupby(level=[0,1]).min().reset_index()
    wk_min = wk_min.rename({col: min_name}, axis='columns')
    temp_df = temp_df.merge(wk_min, on=['pid', 'combined_hash'], how='outer')

    wk_max = wk_counts.groupby(level=[0,1]).max().reset_index()
    wk_max = wk_max.rename({col: max_name}, axis='columns')
    temp_df = temp_df.merge(wk_max, on=['pid', 'combined_hash'], how='outer')


    final_features = [mean_name, std_name, min_name, med_name, max_name]
    comm_features[final_features] = temp_df[final_features]

    return comm_features


def build_intensity_features(comm_features, call_df, sms_df):
    """Returns feature_df with intensity features extracted from call_df, sms_df.
    
    Should be called after build_count_features

    Features created:
    - {mean, std} {out, in} {call, sms} per study day
    """

    # for week groupings 
    group_key = ['pid', 'combined_hash', pd.Grouper(key='date_days', freq='W'), 'comm_direction']

    call_group = call_df.groupby(group_key, 
                                 as_index=False).size().unstack(level=-1, fill_value=0)
    # call_group = call_df.groupby(['pid', 'combined_hash', 'date_days', 'comm_direction'], 
    #                              as_index=False).size().unstack(level=-1, fill_value=0)
    call_group = call_group.reset_index()

    if 'INCOMING' in call_group.columns:
        comm_features = intensity_helper(comm_features, call_group, 'INCOMING', 'in_call')
    if 'OUTGOING' in call_group.columns:
        comm_features = intensity_helper(comm_features, call_group, 'OUTGOING', 'out_call')

    sms_group = sms_df.groupby(group_key, 
                               as_index=False).size().unstack(level=-1, fill_value=0)
    sms_group = sms_group.reset_index()

    if 'INCOMING' in sms_group.columns:
        comm_features = intensity_helper(comm_features, sms_group, 'INCOMING', 'in_sms')
    
    if 'OUTGOING' in sms_group.columns:
        comm_features = intensity_helper(comm_features, sms_group, 'OUTGOING', 'out_sms')
    
    return comm_features    
    

def temporal_tendency_helper(df, group_col, comm_label, norm=None):
    """Convenience function for extracting temporal tendency features.
    
    column names will be (group_col + temporal_index + comm_label).

    norm is an optional pd.Series to be used for normalization, otherwise
    default to the summed count of the given df.

    Returns a df with the extracted features

    TODO add parameters to handle different normalizations, target counts
    """
    temp_tendency_df = df.groupby(['pid', 'combined_hash', group_col], 
                                  as_index=False).size().unstack(level=-1, fill_value=0)
    #cols = [x for x in range(len(temp_tendency_df.columns.values))]
    cols = list(temp_tendency_df.columns.values)
    temp_tendency_df = temp_tendency_df.reset_index()
    
    if norm is None:
        norm = temp_tendency_df[cols].sum(axis=1)

    temp_tendency_df[cols] = temp_tendency_df[cols].div(norm, axis=0)
    temp_tendency_df.set_index(['pid', 'combined_hash'], inplace=True)
    temp_tendency_df.rename(columns=lambda x: group_col + '_' + str(x) + '_' + comm_label, inplace=True)
    temp_tendency_df = temp_tendency_df.reset_index()
    
    # fill divide by 0's with 0, TODO not propogating through after pd.merge()
    # temp_tendency_df = temp_tendency_df.fillna(0)

    return temp_tendency_df


def duration_temporal_helper(call_df, group_col, comm_label):
    """Call duration temporal helper, due to aggregation and normalization differences.

    """
    temp_tendency_df = call_df.groupby(['pid', 'combined_hash', group_col])['call_duration'].sum().unstack(level=-1, fill_value=0)
    # cols = [x for x in range(len(temp_tendency_df.columns.values))]
    cols = list(temp_tendency_df.columns.values)
    
    tot_calls = call_df.groupby(['pid', 'combined_hash']).count()['call_duration']
    temp_tendency_df[cols] = temp_tendency_df[cols].div(tot_calls, axis=0)
    temp_tendency_df = temp_tendency_df.reset_index()
    temp_tendency_df.set_index(['pid', 'combined_hash'], inplace=True)
    temp_tendency_df.rename(columns=lambda x: group_col + '_' + str(x) + '_' + comm_label, inplace=True)
    temp_tendency_df = temp_tendency_df.reset_index()
    
    return temp_tendency_df


def build_temporal_features(comm_features, call_df, sms_df):
    """Returns comm_features with temporal tendency features.

    Features created over time_of_day_{0-3} and day_{0-6}:
    - {call, sms, comm}: # {call, sms, comm} at time / total {call, sms, comm}
    - {call_dur}: sum call_duration at time / total_calls
    - {long_call, miss_call}_{out, in}: # {length calls, missed calls} at time / {outgoing, incoming} calls
    - call_select: # calls at time / total comms
    - out_comm: # outgoing comms at time / total comms
    """
    feature_dfs = []

    # calls
    feature_dfs.append(temporal_tendency_helper(call_df, 'time_of_day', 'call'))
    feature_dfs.append(temporal_tendency_helper(call_df, 'day', 'call'))

    # sms
    feature_dfs.append(temporal_tendency_helper(sms_df, 'time_of_day', 'sms'))
    feature_dfs.append(temporal_tendency_helper(sms_df, 'day', 'sms'))

    # duration
    feature_dfs.append(duration_temporal_helper(call_df, 'time_of_day', 'call_dur'))
    feature_dfs.append(duration_temporal_helper(call_df, 'day', 'call_dur'))

    # comms
    comm_df = call_df.append(sms_df)
    feature_dfs.append(temporal_tendency_helper(comm_df, 'time_of_day', 'comm'))
    feature_dfs.append(temporal_tendency_helper(comm_df, 'day', 'comm'))
    # out comms
    out_comm = comm_df.loc[comm_df['comm_direction'] == 'OUTGOING']
    tot_comms = comm_df.groupby(['pid', 'combined_hash'], 
                                 as_index=False).count()['comm_direction']
    feature_dfs.append(temporal_tendency_helper(out_comm, 'time_of_day', 'comm_out', tot_comms))
    feature_dfs.append(temporal_tendency_helper(out_comm, 'day', 'comm_out', tot_comms))    
    
    # missed/lengthy calls 
    miss_df = call_df.loc[call_df['comm_direction'] == 'MISSED']
    long_df = call_df.loc[call_df['call_duration'] > call_df['call_duration'].median()]
    df_d = {'miss_call': miss_df, 'long_call': long_df}

    out_df = call_df.loc[call_df['comm_direction'] == 'OUTGOING']
    out_calls = out_df.groupby(['pid', 'combined_hash'], 
                               as_index=False).count()['comm_direction']
    in_df = call_df.loc[call_df['comm_direction'] == 'INCOMING']
    in_calls = in_df.groupby(['pid', 'combined_hash'], 
                              as_index=False).count()['comm_direction']
    norm_d = {'out': out_calls, 'in': in_calls}

    for norm_name, norm in norm_d.items():
        for df_name, df in df_d.items():
            col_name = df_name + '_' + norm_name
            feature_dfs.append(temporal_tendency_helper(df, 'time_of_day', col_name, norm))
            feature_dfs.append(temporal_tendency_helper(df, 'day', col_name, norm))

    # merge all temporal features
    for df in feature_dfs:
        comm_features = comm_features.merge(df, on=['pid', 'combined_hash'], how='outer')

    # for some reason, merge converts the zeros into nans
    # comm_features = comm_features.fillna(0)

    return comm_features


def build_channel_selection_features(comm_features, raw_df):
    """Returns comm_features with channel selection features.

    Features created:
    - out_comm: out comm / total comm
    - call_tendency: call count / total comm
    """
    comm_group = raw_df.groupby(['pid', 'combined_hash', 'comm_direction'], 
                                as_index=False).size().unstack(level=-1, fill_value=0)
    comm_group = comm_group.reset_index()
    
    temp_df = comm_features.merge(comm_group[['pid', 'combined_hash','OUTGOING']], 
                                  on=['pid', 'combined_hash'], 
                                  how='outer')
    comm_features['out_comm'] = temp_df['OUTGOING'] / temp_df['total_comms']

    comm_features['call_tendency'] = comm_features['total_calls'] / comm_features['total_comms']

    return comm_features


def build_avoidance_features(comm_features, call_df, sms_df):
    """Returns comm_features with avoidance features.

    Features created:
    - missed_{in, out}_calls: missed call / {in, out} calls
    - in_out_sms: in texts / out texts
    """
    call_group = call_df.groupby(['pid', 'combined_hash', 'comm_direction'], 
                                 as_index=False).size().unstack(level=-1, fill_value=0)
    call_group = call_group.reset_index()
    
    if ('INCOMING' in call_group.columns) and ('MISSED' in call_group.columns):
        call_group['missed_in_calls'] = call_group['MISSED'] / call_group['INCOMING']
    else:
        call_group['missed_in_calls'] = np.nan

    if ('OUTGOING' in call_group.columns) and ('MISSED' in call_group.columns):
        call_group['missed_out_calls'] = call_group['MISSED'] / call_group['OUTGOING']
    else:
        call_group['missed_out_calls'] = np.nan

    sms_group = sms_df.groupby(['pid', 'combined_hash', 'comm_direction'], 
                               as_index=False).size().unstack(level=-1, fill_value=0)
    sms_group = sms_group.reset_index()

    if ('OUTGOING' in sms_group.columns) and ('INCOMING' in sms_group.columns):
        sms_group['in_out_sms'] = sms_group['INCOMING'] / sms_group['OUTGOING']
    else:
        sms_group['in_out_sms'] = np.nan
    
    comm_features = comm_features.merge(
        call_group[['pid', 'combined_hash','missed_in_calls', 'missed_out_calls']], 
        on=['pid', 'combined_hash'], 
        how='outer')
    
    comm_features = comm_features.merge(
        sms_group[['pid', 'combined_hash','in_out_sms']], 
        on=['pid', 'combined_hash'], 
        how='outer')

    comm_features = comm_features.replace([np.inf, -np.inf], np.nan)
    
    return comm_features
    

def build_duration_features(comm_features, call_df):
    """Builds features associated with call duration.

    Features created:
    - {avg, med, max}_{in, out}_duration
    - tot_call_duration: sum of all call duration
    - tot_long_calls: number of lengthy calls (double the median population len)
    """
    call_dur = 'call_duration'

    direction_tup = [('in', 'INCOMING'), ('out', 'OUTGOING')]

    # in/out features
    for name, comm_dir in direction_tup:
        avg_col = "avg_{}_duration".format(name)
        max_col = "max_{}_duration".format(name)
        med_col = "med_{}_duration".format(name)

        dir_call_df = call_df.loc[call_df['comm_direction'] == comm_dir]

        avg_dur = dir_call_df.groupby(['pid', 'combined_hash'], as_index=False)[call_dur].mean()
        avg_dur = avg_dur.rename({call_dur: avg_col}, axis='columns')

        max_dur = dir_call_df.groupby(['pid', 'combined_hash'], as_index=False)[call_dur].max()
        max_dur = max_dur.rename({call_dur: max_col}, axis='columns')
        
        med_dur = dir_call_df.groupby(['pid', 'combined_hash'], as_index=False)[call_dur].median()
        med_dur = med_dur.rename({call_dur: med_col}, axis='columns')

        comm_features = comm_features.merge(avg_dur, on=['pid', 'combined_hash'], how='outer')
        comm_features = comm_features.merge(max_dur, on=['pid', 'combined_hash'], how='outer')
        comm_features = comm_features.merge(med_dur, on=['pid', 'combined_hash'], how='outer')

    # total duration
    tot_dur = call_df.groupby(['pid', 'combined_hash'], as_index=False)[call_dur].sum()
    tot_dur = tot_dur.rename({call_dur: "tot_call_duration"}, axis='columns')
    comm_features = comm_features.merge(tot_dur, on=['pid', 'combined_hash'], how='outer')

    # lengthy features
    pop_median = call_df[call_dur].median()
    long_calls = call_df.loc[call_df[call_dur] > pop_median]
    long_calls = long_calls.groupby(['pid', 'combined_hash'], as_index=False)[call_dur].count()
    long_calls = long_calls.rename({call_dur: 'tot_long_calls'}, axis='columns')
    comm_features = comm_features.merge(long_calls, on=['pid', 'combined_hash'], how='outer')

    return comm_features


def maintenance_features_helper(group_df, col_name, lookback, norm_df):
    """Helper function to compute last lookback week counts.

    Returns a df with the aggregated counts with column {col_name}_last_{lookback}_wks
    """
    group_df = group_df.reset_index()
    target_col = group_df.columns.values[-1] # last col is the target col
    out_col = col_name + "_last_" + str(lookback) + "_wks" 
    
    comm_wks = group_df.groupby(['pid', 'combined_hash'], as_index=False).tail(lookback)
    comm_wks = comm_wks.groupby(['pid', 'combined_hash'], as_index=False)[target_col].sum()
    comm_wks = comm_wks.rename({target_col: out_col}, axis='columns')
    
    norm_col = norm_df.columns.values[-1]

    comm_wks = comm_wks.merge(norm_df, on=['pid', 'combined_hash'], how='outer')
    comm_wks[out_col] = comm_wks[out_col] / comm_wks[norm_col]
    
    return comm_wks[['pid', 'combined_hash', out_col]]


def build_maintenance_features(comm_features, call_df, sms_df):
    """Builds maintenance cost features for call, sms, comm.

    Features created:
    - {call, sms, comm}_last_{2, 6}_wks: # {call, sms, comms} over the past {2,6} weeks / total {call, sms, comms}
    - call_dur_last{2, 6}_wks: sum of call duration over the past {2,6} weeks / total calls
    """
    group_key = ['pid', 'combined_hash', pd.Grouper(key='date_days', freq='W')]

    feature_dfs = []

    # calls
    call_group = call_df.groupby(group_key)['contact_type'].count()
    tot_calls = comm_features[['pid', 'combined_hash', 'total_calls']]
    feature_dfs.append(maintenance_features_helper(call_group, 'call', 2, tot_calls))
    feature_dfs.append(maintenance_features_helper(call_group, 'call', 6, tot_calls))

    # sms
    sms_group = sms_df.groupby(group_key)['contact_type'].count()
    tot_sms = comm_features[['pid', 'combined_hash', 'total_sms']]
    feature_dfs.append(maintenance_features_helper(sms_group, 'sms', 2, tot_sms))
    feature_dfs.append(maintenance_features_helper(sms_group, 'sms', 6, tot_sms))

    # duration
    dur_group = call_df.groupby(group_key)['call_duration'].sum()
    feature_dfs.append(maintenance_features_helper(dur_group, 'call_dur', 2, tot_calls))
    feature_dfs.append(maintenance_features_helper(dur_group, 'call_dur', 6, tot_calls))
    
    # total comms
    comm_df = call_df.append(sms_df)
    comm_group = comm_df.groupby(group_key)['contact_type'].count()
    tot_comms = comm_features[['pid', 'combined_hash', 'total_comms']]
    feature_dfs.append(maintenance_features_helper(comm_group, 'comm', 2, tot_comms))
    feature_dfs.append(maintenance_features_helper(comm_group, 'comm', 6, tot_comms))

    for df in feature_dfs:
        comm_features = comm_features.merge(df, on=['pid', 'combined_hash'], how='outer')
    
    return comm_features


class HolidayCalendar(AbstractHolidayCalendar):
    """Custom holiday calendar to match Min et al.
    """
    rules = [
        Holiday('Christmas', month=12, day=25),
        Holiday('Valentines', month=2, day=14),
        Holiday('NewYears', month=1, day=1),
        USThanksgivingDay
    ]


def filter_by_holiday(comm_df):
    """Filters the given df by entries that occur on a holiday.
    """
    cal = HolidayCalendar()
    start_date = comm_df['date_days'].min()
    end_date = comm_df['date_days'].max()
    holidays = cal.holidays(start=start_date, end=end_date)

    return comm_df.loc[comm_df['date_days'].isin(holidays)]


def build_holiday_features(comm_features, comm_df):
    """Builds holiday communication frequency features.
    
    Holidays as defined by Min et al are:
    - Thanksgiving
    - Christmas
    - New Year's Day
    - Valentine's Day

    features created:
    - holiday_comms: # outgoing communications on holidays / total communications
    """
    holiday_col = 'holiday_comms'
    holiday_comms = filter_by_holiday(comm_df)
    holiday_out_comms = holiday_comms.loc[holiday_comms['comm_direction'] == 'OUTGOING']
    holiday_counts = holiday_out_comms.groupby(['pid', 'combined_hash'], 
                                               as_index=False)['comm_direction'].count()
    holiday_counts = holiday_counts.rename({'comm_direction': holiday_col}, axis='columns')

    comm_features = comm_features.merge(holiday_counts, on=['pid', 'combined_hash'], how='outer')
    comm_features[holiday_col] = comm_features[holiday_col].fillna(0)
    comm_features[holiday_col] = comm_features[holiday_col] / comm_features['total_comms']

    return comm_features


def build_nan_features(comm_features, fill_val=0):
    """Adds additional feature columns for nans and fills NaNs with fill_val.

    features created:
    - {col}_nan_indicator: one-hot column indicating whether 
    """
    comm_indicator = comm_features.isnull().astype(int).add_suffix("_nan_indicator")
    # keep indicator cols that correspond to cols with NaNs
    comm_indicator = comm_indicator.loc[:, (comm_indicator.sum(axis=0) > 0)]

    indicator_cols = comm_indicator.columns
    comm_features[indicator_cols] = comm_indicator[indicator_cols]

    if fill_val == 'mean':
        print('filling mean')
        fill_val = comm_features.mean()
    if fill_val == 'median':
        print('filling median')
        fill_val = comm_features.median()
    
    comm_features = comm_features.fillna(fill_val)

    return comm_features
    

def build_emc_features(comm_features, comm_df, emc_df, hash_dict, pr_dict):
    """Adds emc_feature columns to comm_features.

    """
    emc_df = emc_df.reset_index(drop=True)

    canonical_dict = {}
    for k,v in hash_dict.items():
        if len(v) > 0:
            canonical_dict[k] = v[0]

    emc_dict = {}
    emc_features = ['q1_want', 'q2_talk', 'q3_loan', 'q4_closeness']
    for i in range(emc_df.shape[0]):
        hash_name = emc_df.loc[i, 'contact_name']
        if hash_name in hash_dict:
            canonical_md5 = hash_dict[hash_name]
            if len(canonical_md5) > 0 and canonical_md5[0] in pr_dict:
                non_canonical_hashes = pr_dict[canonical_md5[0]]
                for h in non_canonical_hashes:
                    emcs = {}
                    for col in emc_features:
                        emcs[col] = emc_df.loc[i, col]
                    emc_dict[h] = emcs      

    emc_df['canonical_hash'] = emc_df['contact_name'].map(canonical_dict)
    comm_df['emc_dict'] = comm_df['contact_number'].map(emc_dict)
    combined_to_emc =  pd.Series(comm_df['emc_dict'].values,index=comm_df['combined_hash']).to_dict()

    emc_features = pd.DataFrame(index=combined_to_emc.keys())
    for combined_hash, q_dict in combined_to_emc.items():
        for k,v in q_dict.items():
            emc_features.loc[combined_hash, k] = v
    emc_features.index.rename("combined_hash", inplace=True)
    emc_features = emc_features.reset_index()

    comm_features = comm_features.merge(emc_features, how='outer', on=['combined_hash'])

    return comm_features


def build_demo_features(comm_df, demo_df, age_gender_only=True):
    """Adds demographic features of the egos to the feature frame.

    Defaults to adding only age and gender.

    TODO how to handle ordinal variables?
    """
    demo_cols = ['age', 'gender', 'education', 'employment', 'live_together', 'race', 'ethnicity', 'marital_status']
    if age_gender_only:
        demo_cols = ['age', 'gender']

    for demo in demo_cols:
        demo_dict = pd.Series(demo_df[demo].values,index=demo_df['pid']).to_dict()
        col_name = "ego_{}".format(demo)
        comm_df[col_name] = comm_df['pid'].map(demo_dict) 
        if demo != 'age':
            # change everything to one-hot
            # if demo == 'education':
            #     edu_dict = {
            #         'some_hs': 0,
            #         'completed_hs': 1,
            #         'some_college': 2,
            #         'associates': 3, 
            #         'bachelors': 4,
            #         'masters': 5,
            #         'pro_doctoral': 6 
            #     }
            #     comm_df[col_name] = comm_df[col_name].map(edu_dict)

            # elif demo == 'live_together':
            #     live_dict = {
            #         'alone': 0, 
            #         '1_other': 1,
            #         '2_others': 2, 
            #         '>=3_others': 3
            #     }
            #     comm_df[col_name] = comm_df[col_name].map(live_dict)

            # else:
            
            # tile dummy variables out for non-ordinal categorical variables
            comm_df = pd.get_dummies(comm_df, columns=[col_name])

    return comm_df


visit_reasons = [
    'visit_reason:entertainment',
    'visit_reason:errand',
    'visit_reason:home',
    'visit_reason:work',
    'visit_reason:exercise',
    'visit_reason:dining',
    'visit_reason:socialize',
    'visit_reason:travel/traffic',
    'visit_reason:other'
]


locations = [
    "loc:home",
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

def build_location_features(comm_features, comm_df):
    """Adds semantic location features to the feature frame.

    Assumes the incoming comm_df dataframe has the semantic location features 
    populated.

    TODO refactor with call_df, sms_df as parameters?
    TODO should we divide by total number of communications?
    """

    call_df = comm_df.loc[comm_df['comm_type'] == 'PHONE']
    sms_df = comm_df.loc[comm_df['comm_type'] == 'SMS']

    # call features
    call_visit = call_df.groupby(['pid', 'combined_hash'])[visit_reasons].sum()
    call_visit[visit_reasons] = call_visit[visit_reasons].divide(call_visit.sum(axis=1), axis='rows')
    call_visit = call_visit.add_prefix('call_')
    call_visit = call_visit.reset_index()

    call_loc = call_df.groupby(['pid', 'combined_hash'])[locations].sum()
    call_loc[locations] = call_loc[locations].divide(call_loc.sum(axis=1), axis='rows')
    call_loc = call_loc.add_prefix('call_')
    call_loc = call_loc.reset_index()

    #sms features
    sms_visit = sms_df.groupby(['pid', 'combined_hash'])[visit_reasons].sum()
    sms_visit[visit_reasons] = sms_visit[visit_reasons].divide(sms_visit.sum(axis=1), axis='rows')
    sms_visit = sms_visit.add_prefix('sms_')
    sms_visit = sms_visit.reset_index()

    sms_loc = sms_df.groupby(['pid', 'combined_hash'])[locations].sum()
    sms_loc[locations] = sms_loc[locations].divide(sms_loc.sum(axis=1), axis='rows')
    sms_loc = sms_loc.add_prefix('sms_')
    sms_loc = sms_loc.reset_index()

    comm_features = comm_features.merge(call_visit, on=['pid', 'combined_hash'], how='outer')
    comm_features = comm_features.merge(call_loc, on=['pid', 'combined_hash'], how='outer')
    comm_features = comm_features.merge(sms_visit, on=['pid', 'combined_hash'], how='outer')
    comm_features = comm_features.merge(sms_loc, on=['pid', 'combined_hash'], how='outer')
    
    return comm_features

