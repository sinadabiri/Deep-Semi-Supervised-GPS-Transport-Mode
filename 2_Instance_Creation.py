import numpy as np
import pickle
from geopy.distance import vincenty
import math
import time
import random
import pandas as pd

# Change the current working directory to the location of 'Combined Trajectory_Label_Geolife' folder.

current = time.clock()
min_threshold = 20
max_threshold = 248
min_distance = 150
min_time = 60


filename = '../Mode-codes-Revised/paper2_Trajectory_Label.pickle'
with open(filename, 'rb') as f:
    trajectory_all_user_with_label, trajectory_all_user_wo_label = pickle.load(f)


# trajectory_all_user_with_label = trajectory_all_user_with_label[:2]
# trajectory_all_user_wo_label = trajectory_all_user_wo_label[:2]


# Identify the Speed and Acceleration limit
SpeedLimit = {0: 7, 1: 12, 2: 120./3.6, 3: 180./3.6, 4: 120/3.6}
# Online sources for Acc: walk: 1.5 Train 1.15, bus. 1.25 (.2), bike: 2.6, train:1.5
AccLimit = {0: 3, 1: 3, 2: 2, 3: 10, 4: 3}


def unlabeled_gps_to_trip(trajectory_one_user, trip_time):
    """
    This function divide total GPS trajectory of one user into some trips, when the travel time between two consecutive
    GPS points exceeds the "trip time". Also, remove the erroneous GPS points that their travel time is non-positive.
    :param trajectory_one_user: A sequence of a users' all GPS points.
    :param trip_time: the maximum time for dividing a GPS sequence into trips.
    :return: a user's  trips
    """
    trip = []
    all_trip_one_user = []
    i = 0
    while i < len(trajectory_one_user) - 1:
        delta_time = (trajectory_one_user[i+1][2] - trajectory_one_user[i][2]) * 24 * 3600
        if 0 < delta_time <= trip_time:
            trip.append(trajectory_one_user[i])
            i += 1
        elif delta_time > trip_time:
            trip.append(trajectory_one_user[i])
            all_trip_one_user.append(trip)
            trip = []
            i += 1
        else:
            trajectory_one_user.remove(trajectory_one_user[i + 1])
    return all_trip_one_user


def labeled_gps_to_trip(trajectory_one_user, trip_time):
    """
    This function divides total labeled-GPS trajectory of one user into some trips, when either the travel time between
    two consecutive GPS points exceeds the "trip time" or the mode changes.
    Also, remove the erroneous GPS points that their travel time is non-positive.
    :param trajectory_one_user: A sequence of a users' all GPS points.
    :param trip_time: the maximum time for dividing a GPS sequence into trips.
    :return: a user's  trips
    """
    trip = []
    all_trip_one_user = []
    i = 0
    while i < len(trajectory_one_user) - 1:
        delta_time = (trajectory_one_user[i+1][2] - trajectory_one_user[i][2]) * 24 * 3600
        mode_not_change = (trajectory_one_user[i+1][3] == trajectory_one_user[i][3])
        if 0 < delta_time <= trip_time and mode_not_change:
            trip.append(trajectory_one_user[i])
            i += 1
        elif delta_time > trip_time or not mode_not_change:
            trip.append(trajectory_one_user[i])
            all_trip_one_user.append(trip)
            trip = []
            i += 1
        elif delta_time <= 0:
            trajectory_one_user.remove(trajectory_one_user[i + 1])
    return all_trip_one_user

# The two following lists contain all trips of all users.
trip_all_user_with_label = [labeled_gps_to_trip(trajectory, trip_time=20*60) for trajectory in
                            trajectory_all_user_with_label]
trip_all_user_wo_label = [unlabeled_gps_to_trip(trajectory, trip_time=20*60) for trajectory in
                          trajectory_all_user_wo_label]


def compute_delta_time(p1, p2):
    """
    :param p2: trajectory_one_user[i + 1]
    :param p1: trajectory_one_user[i]
    :return:
    """
    return (p2[2] - p1[2]) * 24 * 3600


def compute_distance(p1, p2):
    lat_long_1 = (p1[0], p1[1])
    lat_long_2 = (p2[0], p2[1])
    return vincenty(lat_long_1, lat_long_2).meters


def compute_speed(distance, delta_time):
    return distance/delta_time


def compute_acceleration(speed1, speed2, delta_time):
    return (speed2 - speed1) / delta_time


def compute_jerk(acc1, acc2, delta_time):
    return (acc2 - acc1) / delta_time


def compute_bearing(p1, p2):
    y = math.sin(math.radians(p2[1]) - math.radians(p1[1])) * math.radians(math.cos(p2[0]))
    x = math.radians(math.cos(p1[0])) * math.radians(math.sin(p2[0])) - \
        math.radians(math.sin(p1[0])) * math.radians(math.cos(p2[0])) \
        * math.radians(math.cos(p2[1]) - math.radians(p1[1]))
    # Convert radian from -pi to pi to [0, 360] degree
    return (math.atan2(y, x) * 180. / math.pi + 360) % 360


def compute_bearing_rate(bearing1, bearing2):
    return abs(bearing1 - bearing2)


def remove_error_labeled(trip_motion_features, mode):
    outlier_speed = [index for index, item in enumerate(trip_motion_features[3]) if item > SpeedLimit[mode]]
    outlier_acc = [index for index, item in enumerate(trip_motion_features[4]) if
                   abs(item) > AccLimit[mode] and index not in outlier_speed]
    outlier = outlier_speed + outlier_acc
    trip_motion_features = np.delete(np.array(trip_motion_features), outlier, axis=1)
    return trip_motion_features


def remove_error_unlabeled(trip_motion_features):
    speed = trip_motion_features[3]
    upper_quartile = np.percentile(speed, 75)
    lower_quartile = np.percentile(speed, 25)
    iqr = (upper_quartile - lower_quartile) * 1.5
    quartile_set = (max(lower_quartile - iqr, 0), min(upper_quartile + iqr, 50))
    outlier_speed = [index for index, item in enumerate(speed) if item < quartile_set[0] or
                     item > quartile_set[1]]
    acc = trip_motion_features[4]
    upper_quartile = np.percentile(acc, 75)
    lower_quartile = np.percentile(acc, 25)
    iqr = (upper_quartile - lower_quartile) * 1.5
    quartile_set = (max(lower_quartile - iqr, -10), min(upper_quartile + iqr, 10))
    outlier_acc = [index for index, item in enumerate(acc) if
                   (item < quartile_set[0] or item > quartile_set[1]) and index not in outlier_speed]
    outlier = outlier_speed + outlier_acc
    trip_motion_features = np.delete(np.array(trip_motion_features), outlier, axis=1)
    return trip_motion_features


def compute_trip_motion_features(all_trip_one_user, data_type):
    """
    This function computes the motion features for every trip (i.e., a sequence of GPS points).
    There are four types of motion features: speed, acceleration, jerk, and bearing rate.
    :param trip: a sequence of GPS points
    :param data_type: is it related to a 'labeled' and 'unlabeled' data set.
    :return: A list with four sub-lists, where every sub-list is a motion feature.
    """
    all_trip_motion_features_one_user = []
    for trip in all_trip_one_user:
        if len(trip) >= 4:
            relative_distance = []
            delta_time = []
            relative_speed = []
            speed = []
            acc = []
            jerk = []
            bearing_rate = []
            delta_time_1 = compute_delta_time(trip[0], trip[1])
            distance_1 = compute_distance(trip[0], trip[1])
            speed1 = compute_speed(distance_1, delta_time_1)
            delta_time_2 = compute_delta_time(trip[1], trip[2])
            distance_2 = compute_distance(trip[1], trip[2])
            speed2 = compute_speed(distance_2, delta_time_2)
            acc1 = compute_acceleration(speed1, speed2, delta_time_1)
            for i in range(len(trip) - 3):
                delta_time_1 = compute_delta_time(trip[i], trip[i + 1])
                delta_time_3 = compute_delta_time(trip[i + 2], trip[i + 3])
                distance_3 = compute_distance(trip[i + 2], trip[i + 3])
                speed3 = compute_speed(distance_3, delta_time_3)
                acc2 = compute_acceleration(speed2, speed3, delta_time_2)
                relative_distance.append(distance_1)
                delta_time.append(delta_time_1)
                relative_speed.append(abs((speed2 - speed1)) / speed1 if speed1 != 0 else 0)
                speed.append(speed1)
                acc.append(acc1)
                jerk.append(compute_jerk(acc1, acc2, delta_time=delta_time_1))
                bearing_rate.append(compute_bearing_rate(compute_bearing(trip[i], trip[i + 1]),
                                                         compute_bearing(trip[i + 1], trip[i + 2])))
                delta_time_2 = delta_time_3
                distance_1 = distance_2
                distance_2 = distance_3
                speed1 = speed2
                speed2 = speed3
                acc1 = acc2
            trip_motion_features = [relative_distance, delta_time, relative_speed, speed, acc, jerk, bearing_rate]
            if data_type == 'labeled':
                mode = trip[0][3]
                # Randomly check that all trip[i][3] have the same mode
                assert trip[0][3] == trip[np.random.randint(1, len(trip)-1, 1)[0]][3]
                trip_motion_features = remove_error_labeled(trip_motion_features, mode)
                all_trip_motion_features_one_user.append((trip_motion_features, mode))
            if data_type == 'unlabeled':
                trip_motion_features = remove_error_unlabeled(trip_motion_features)
                all_trip_motion_features_one_user.append(trip_motion_features)
    return all_trip_motion_features_one_user

trip_motion_all_user_with_label = [compute_trip_motion_features(user, data_type='labeled') for user
                                   in trip_all_user_with_label]
trip_motion_all_user_wo_label = [compute_trip_motion_features(user, data_type='unlabeled') for user
                                 in trip_all_user_wo_label]

# This pickling and unpickling is due to large computation time before this line.
with open('paper2_trips_motion_features_temp.pickle', 'wb') as f:
    pickle.dump([trip_motion_all_user_with_label, trip_motion_all_user_wo_label], f)

filename = '../Mode-codes-Revised/paper2_trips_motion_features_temp.pickle'
with open(filename, 'rb') as f:
    trip_motion_all_user_with_label, trip_motion_all_user_wo_label = pickle.load(f)


def trip_check_thresholds(trip_motion_all_user, min_threshold, min_distance, min_time, data_type):
    # Remove trip with less than a min GPS point, less than a min-distance, less than a min trip time.
    all_user = []
    if data_type == 'labeled':
        for user in trip_motion_all_user:
            all_user.append(list(filter(lambda trip: len(trip[0][0]) >= min_threshold and np.sum(trip[0][0, :]) >= min_distance
                                          and np.sum(trip[0][1, :]) >= min_time, user)))
    if data_type == 'unlabeled':
        for user in trip_motion_all_user:
            all_user.append(list(filter(lambda trip: len(trip[0]) >= min_threshold and np.sum(trip[0, :]) >= min_distance
                                          and np.sum(trip[1, :]) >= min_time, user)))
    return all_user

# Apply the threshold values to each GPS segment
trip_motion_all_user_with_label = trip_check_thresholds(trip_motion_all_user_with_label, min_threshold=min_threshold, min_distance=min_distance, min_time=min_time,
                                                        data_type='labeled')
trip_motion_all_user_wo_label = trip_check_thresholds(trip_motion_all_user_wo_label, min_threshold=min_threshold, min_distance=min_distance, min_time=min_time,
                                                        data_type='unlabeled')

# Find the median size (M) as the fixed size of all GPS segments.
trip_length_labeled = [len(trip[0][0]) for user in trip_motion_all_user_with_label for trip in user]
trip_length_unlabeled = [len(trip[0]) for user in trip_motion_all_user_wo_label for trip in user]

print('Descriptive statistics for labeled',  pd.Series(trip_length_labeled).describe(percentiles=[0.05, 0.1, 0.15,
                                                                                                      0.25, 0.5, 0.75,
                                                                                                      0.85, 0.9, 0.95]))
print('Descriptive statistics for unlabeled',  pd.Series(trip_length_unlabeled).describe(percentiles=[0.05, 0.1, 0.15,
                                                                                                      0.25, 0.5, 0.75,
                                                                                                      0.85, 0.9, 0.95]))
'''
# Now, we have all trips in a list from all users. So time to Create train, test, and validation sets
train_trip_motion_all_user_with_label = []
val_trip_motion_all_user_with_label = []
test_trip_motion_all_user_with_label = []
for user in trip_motion_all_user_with_label:
    random.shuffle(user)
    length = len(user)
    train_trip_motion_all_user_with_label.extend(user[:round(0.7*length)])
    val_trip_motion_all_user_with_label.extend(user[round(0.7*length):round(0.8*length)])
    test_trip_motion_all_user_with_label.extend(user[round(0.8*length):])
'''
# Put trips of all user together.
trip_motion_all_user_with_label = [trip for user in trip_motion_all_user_with_label for trip in user]
random.shuffle(trip_motion_all_user_with_label)
trip_motion_all_user_wo_label = [trip for user in trip_motion_all_user_wo_label for trip in user]
random.shuffle(trip_motion_all_user_wo_label)


with open('paper2_trips_motion_features_NotFixedLength_woOutliers.pickle', 'wb') as f:
    pickle.dump([trip_motion_all_user_with_label, trip_motion_all_user_wo_label], f)

print('Running time', time.clock() - current)
