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

DAY_DIVISOR = 4  # 6 hour chunks

def comm_feature_extract(comm_df, ema_df):
    """Generates a DataFrame with extracted comm features, one contact per row

    params:
    df: an input DataFrame as described above

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
    comm_features = build_channel_selection_features(comm_features, comm_df)
    comm_features = build_avoidance_features(comm_features, call_df, sms_df)

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

    # build total logged days for each participant
    total_recorded_days = ema_df.groupby('pid')['date'].nunique()
    comm_features['total_days'] = comm_features.apply(lambda x: total_recorded_days[x.pid], 
                                                      axis=1)

    comm_features['reg_call'] = comm_features['total_call_days'] / comm_features['total_days']
    comm_features['reg_sms'] = comm_features['total_sms_days'] / comm_features['total_days']
    comm_features['reg_comm'] = comm_features['total_comm_days'] / comm_features['total_days']

    return comm_features


def build_intensity_features(comm_features, raw_df):
    """Returns feature_df with intensity features extracted from raw_df
    TODO look up tiling start date/end date
    TODO min/max date within groupbys?

    Features created:
    - {avg, std} {out, in} {call, sms} per study day
    """
    pass
    

def temporal_tendency_helper(df, group_col, comm_label):
    """Convenience function for extracting temporal tendency features.
    column names will be (group_col + temporal_index + comm_label)
    Returns a df with the extracted features
    """
    # TODO figure out indentation
    temp_tendency_df = df.groupby(['pid', 'combined_hash', group_col], 
                                  as_index=False).size().unstack(level=-1, fill_value=0)
    cols = [x for x in range(len(temp_tendency_df.columns.values))]
    temp_tendency_df = temp_tendency_df.reset_index()
    
    tot_calls = temp_tendency_df[cols].sum(axis=1)
    temp_tendency_df[cols] = temp_tendency_df[cols].div(tot_calls, axis=0)
    temp_tendency_df.set_index(['pid', 'combined_hash'], inplace=True)
    temp_tendency_df.rename(columns=lambda x: group_col + '_' + str(x) + '_' + comm_label, inplace=True)
    temp_tendency_df = temp_tendency_df.reset_index()
    
    return temp_tendency_df

def build_temporal_features(comm_features, call_df, sms_df):
    """Returns comm_features with temporal tendency features.

    Features created:
    - # {call, sms} at {time of day, day of week} / total

    """
    time_of_day_calls = temporal_tendency_helper(call_df, 'time_of_day', 'calls')
    day_of_week_calls = temporal_tendency_helper(call_df, 'day', 'calls')

    time_of_day_sms = temporal_tendency_helper(sms_df, 'time_of_day', 'sms')
    day_of_week_sms = temporal_tendency_helper(sms_df, 'day', 'sms')

    comm_features = comm_features.merge(time_of_day_calls, 
                                        on=['pid', 'combined_hash'], 
                                        how='outer')
    comm_features = comm_features.merge(day_of_week_calls, 
                                        on=['pid', 'combined_hash'], 
                                        how='outer')

    comm_features = comm_features.merge(time_of_day_sms, 
                                        on=['pid', 'combined_hash'], 
                                        how='outer')
    comm_features = comm_features.merge(day_of_week_sms, 
                                        on=['pid', 'combined_hash'], 
                                        how='outer')

    return comm_features


def build_channel_selection_features(comm_features, raw_df):
    """Returns comm_features with channel selection features.

    Features created:
    - out comm / total comm
    - call count / total comm

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
    
    return comm_features
    
    


    