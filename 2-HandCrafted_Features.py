import pandas as pd
import numpy as np
import pickle
from geopy.distance import vincenty
import math
import time

# Settings
min_threshold = 20
max_threshold = 248
min_distance = 150  # Meters
min_time = 60  # Seconds
num_class = 5
new_channel = 4
min_percentile = 0
max_percentile = 100
# Change the current working directory to the location of 'Combined Trajectory_Label_Geolife' folder.
current = time.clock()
filename = '../Mode-codes-Revised/paper2_trips_motion_features_NotFixedLength_woOutliers.pickle'
with open(filename, 'rb') as f:
    train_trip_motion_all_user_with_label, val_trip_motion_all_user_with_label, \
    test_trip_motion_all_user_with_label, trip_motion_all_user_wo_label = pickle.load(f)

del trip_motion_all_user_wo_label


def trip_to_fixed_length(trip_motion_all_user, min_threshold, max_threshold, min_distance, min_time, data_type):
    if data_type == 'labeled':
        total_input = []
        total_label = []
        for index, trip in enumerate(trip_motion_all_user):
            trip, mode = trip
            trip_length = len(trip[0])
            if all([trip_length >= min_threshold, trip_length < max_threshold, np.sum(trip[0, :]) >= min_distance,
                    np.sum(trip[1, :]) >= min_time]):
                trip_padded = np.pad(trip, ((0, 0), (0, max_threshold - trip_length)), 'constant')
                total_input.append((trip_padded, mode))
            elif trip_length >= max_threshold:
                    quotient = trip_length // max_threshold
                    for i in range(quotient):
                        trip_truncated = trip[:, i * max_threshold:(i + 1) * max_threshold]
                        if all([np.sum(trip_truncated[0, :]) >= min_distance, np.sum(trip_truncated[1, :]) >= min_time]):
                            total_input.append((trip_truncated, mode))
                    remain_trip = trip[:, (i + 1) * max_threshold:]
                    if all([(trip_length % max_threshold) > min_threshold, np.sum(remain_trip[0, :]) >= min_distance,
                            np.sum(remain_trip[1, :]) >= min_time]):
                        trip_padded = np.pad(remain_trip, ((0, 0), (0, max_threshold - trip_length % max_threshold)),
                                             'constant')
                        total_input.append((trip_padded, mode))

        return total_input

# Max_threshold=200: 200 is the rounded median size of all trips (i.e., GPS trajectory) after removing errors and
# outliers including: 1) max speed and acceleration, (2) trip length less than 10
train_trip_motion_all_user_with_label = trip_to_fixed_length(train_trip_motion_all_user_with_label, min_threshold=min_threshold,
                                                                max_threshold=max_threshold, min_distance=min_distance, min_time=min_time,
                                                                data_type='labeled')
val_trip_motion_all_user_with_label = trip_to_fixed_length(val_trip_motion_all_user_with_label, min_threshold=min_threshold,
                                                                max_threshold=max_threshold, min_distance=min_distance, min_time=min_time,
                                                                data_type='labeled')
test_trip_motion_all_user_with_label = trip_to_fixed_length(test_trip_motion_all_user_with_label, min_threshold=min_threshold,
                                                                max_threshold=max_threshold, min_distance=min_distance, min_time=min_time,
                                                                data_type='labeled')


def create_hand_crafted_features(trip_motion_all_user, set_type):
    # Following lists are hand-crafted features
    Dist = []
    AV = []
    EV = []
    VV = []
    MaxV1 = []
    MaxV2 = []
    MaxV3 = []
    MaxA1 = []
    MaxA2 = []
    MaxA3 = []
    HCR = []  # Heading Change Rate
    SR = []  # Stop Rate
    VCR = []  # Velocity Change Rate
    HC = 19  # Heading rate threshold
    VS = 3.4  # Stop rate threshold
    VR = 0.26  # VCR threshold
    label = []
    for trip in trip_motion_all_user:
        RD = trip[0][0, :]
        DT = trip[0][1, :]
        VC = trip[0][2, :]
        SP = trip[0][3, :]
        AC = trip[0][4, :]
        J = trip[0][5, :]
        BR = trip[0][6, :]

        label.append(trip[1])

        # IN: the instances and number of GPS points in each instance for each user k
        # Basic features
        # Dist: Distance of segments
        Dist.append(np.sum(RD))
        # AV: average velocity
        AV.append(np.sum(RD) / np.sum(DT))
        # EV: expectation velocity
        EV.append(np.mean(SP))
        # VV: variance of velocity
        VV.append(np.var(SP))
        # MaxV1, MaxV2, MaxV3
        sorted_velocity = np.sort(SP)[::-1]
        MaxV1.append(sorted_velocity[0])
        MaxV2.append(sorted_velocity[1])
        MaxV3.append(sorted_velocity[2])
        # MaxA1, MaxA2, MaxA3
        sorted_acceleration = np.sort(AC)[::-1]
        MaxA1.append(sorted_acceleration[0])
        MaxA2.append(sorted_acceleration[1])
        MaxA3.append(sorted_acceleration[2])
        # Heading change rate (HCR)
        Pc = sum(1 for item in list(BR) if item > HC)
        HCR.append(Pc * 1. / np.sum(RD))
        # Stop Rate (SR)
        Ps = sum(1 for item in list(SP) if item < VS)
        SR.append(Ps * 1. / np.sum(RD))
        # Velocity Change Rate (VCR)
        Pv = sum(1 for item in list(VC) if item > VR)
        VCR.append(Pv * 1. / np.sum(RD))

    X = [Dist, AV, EV, VV, MaxV1, MaxV2, MaxV3, MaxA1, MaxA2, MaxA3, HCR, SR, VCR, label]
    X = np.array(X, dtype=np.float32).T

    df = pd.DataFrame(X)
    print(df)
    filename = '2_Hand_Crafted_features_filtered_' + set_type + '.csv'
    df.to_csv(filename, index=False, header=['Distance', 'Average Velocity','Expectation Velocity',
                                                                              'Variance of Velocity',
                                                                              'MaxV1', 'MaxV2', 'MaxV3', 'MaxA1',
                                                                              'MaxA2',
                                                                              'MaxA3', 'Heading Rate Change',
                                                                              'Stop Rate',
                                                                              'Velocity Change Rate', 'Label'])


create_hand_crafted_features(train_trip_motion_all_user_with_label, set_type='train')
create_hand_crafted_features(val_trip_motion_all_user_with_label, set_type='val')
create_hand_crafted_features(test_trip_motion_all_user_with_label, set_type='test')