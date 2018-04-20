import numpy as np
import pickle
import keras

# Import the final output from Instance_creation file, which is the filtered trips for all users.

filename = '../Mode-codes-Revised/paper2_trips_motion_features_NotFixedLength_woOutliers.pickle'
with open(filename, 'rb') as f:
    train_trip_motion_all_user_with_label, val_trip_motion_all_user_with_label, \
    test_trip_motion_all_user_with_label, trip_motion_all_user_wo_label = pickle.load(f)


# Apply some of data preprocessing step in the paper and prepare the final input layer for deep learning

# Settings
# AccLimit = {0: 3, 1: 3, 2: 2, 3: 10, 4: 3}
min_threshold = 20
max_threshold = 248
min_distance = 150  # Meters
min_time = 60  # Seconds
num_class = 5
new_channel = 4
min_percentile = 0
max_percentile = 100


def take_speed_percentile(trip, min_percentile, max_percentile):
    min_threshold = np.percentile(trip[1], min_percentile)
    max_threshold = np.percentile(trip[1], max_percentile)
    index_min = np.where(trip[1] >= min_threshold)[0]
    index_max = np.where(trip[1] <= max_threshold)[0]
    index = np.intersect1d(index_min, index_max)
    trip = trip[:, index]
    return trip


def trip_to_fixed_length(trip_motion_all_user, min_threshold, max_threshold, min_distance, min_time, data_type):
    if data_type == 'labeled':
        total_input = []
        total_label = []
        for index, trip in enumerate(trip_motion_all_user):
            trip, mode = trip
            trip = take_speed_percentile(trip, min_percentile=min_percentile, max_percentile=max_percentile)
            trip_length = len(trip[0])
            if all([trip_length >= min_threshold, trip_length < max_threshold, np.sum(trip[0, :]) >= min_distance,
                    np.sum(trip[1, :]) >= min_time]):
                trip_padded = np.pad(trip, ((0, 0), (0, max_threshold - trip_length)), 'constant')
                total_input.append(trip_padded)
                total_label.append(mode)
            elif trip_length >= max_threshold:
                    quotient = trip_length // max_threshold
                    for i in range(quotient):
                        trip_truncated = trip[:, i * max_threshold:(i + 1) * max_threshold]
                        if all([np.sum(trip_truncated[0, :]) >= min_distance, np.sum(trip_truncated[1, :]) >= min_time]):
                            total_input.append(trip_truncated)
                            total_label.append(mode)
                    remain_trip = trip[:, (i + 1) * max_threshold:]
                    if all([(trip_length % max_threshold) > min_threshold, np.sum(remain_trip[0, :]) >= min_distance,
                            np.sum(remain_trip[1, :]) >= min_time]):
                        trip_padded = np.pad(remain_trip, ((0, 0), (0, max_threshold - trip_length % max_threshold)),
                                             'constant')
                        total_input.append(trip_padded)
                        total_label.append(mode)

        return np.array(total_input), np.array(total_label)

    elif data_type == 'unlabeled':
        total_input = []
        for index, trip in enumerate(trip_motion_all_user):
            trip_length = len(trip[0])
            if all([trip_length >= min_threshold, trip_length < max_threshold, np.sum(trip[0, :]) >= min_distance,
                    np.sum(trip[1, :]) >= min_time]):
                trip_padded = np.pad(trip, ((0, 0), (0, max_threshold - trip_length)), 'constant')
                total_input.append(trip_padded)
            elif trip_length >= max_threshold:
                quotient = trip_length // max_threshold
                for i in range(quotient):
                    trip_truncated = trip[:, i * max_threshold:(i + 1) * max_threshold]
                    if all([np.sum(trip_truncated[0, :]) >= min_distance, np.sum(trip_truncated[1, :]) >= min_time]):
                        total_input.append(trip_truncated)
                remain_trip = trip[:, (i + 1) * max_threshold:]
                if all([trip_length % max_threshold > min_threshold, np.sum(remain_trip[0, :]) >= min_distance,
                        np.sum(remain_trip[1, :]) >= min_time]):
                    trip_padded = np.pad(remain_trip, ((0, 0), (0, max_threshold - trip_length % max_threshold)),
                                         'constant')
                    total_input.append(trip_padded)
        return np.array(total_input)

# Max_threshold=200: 200 is the rounded median size of all trips (i.e., GPS trajectory) after removing errors and
# outliers including: 1) max speed and acceleration, (2) trip length less than 10
Train_X, Train_Y = trip_to_fixed_length(train_trip_motion_all_user_with_label, min_threshold=min_threshold,
                                                                max_threshold=max_threshold, min_distance=min_distance, min_time=min_time,
                                                                data_type='labeled')
Val_X, Val_Y_ori = trip_to_fixed_length(val_trip_motion_all_user_with_label, min_threshold=min_threshold,
                                                                max_threshold=max_threshold, min_distance=min_distance, min_time=min_time,
                                                                data_type='labeled')
Test_X, Test_Y_ori = trip_to_fixed_length(test_trip_motion_all_user_with_label, min_threshold=min_threshold,
                                                                max_threshold=max_threshold, min_distance=min_distance, min_time=min_time,
                                                                data_type='labeled')
X_unlabeled = trip_to_fixed_length(trip_motion_all_user_wo_label, min_threshold=min_threshold,
                                             max_threshold=max_threshold, min_distance=min_distance, min_time=min_time, data_type='unlabeled')


def change_to_new_channel(input):
    input = input[:, 3:, :]
    total_input_new = np.zeros((len(input), 1, max_threshold, new_channel))
    for i in range(len(input)):
        total_input_new[i, 0, :, 0] = input[i, 0, :]
        total_input_new[i, 0, :, 1] = input[i, 1, :]
        total_input_new[i, 0, :, 2] = input[i, 2, :]
        total_input_new[i, 0, :, 3] = input[i, 3, :]
    return total_input_new

Train_X = change_to_new_channel(Train_X)
Train_Y = keras.utils.to_categorical(Train_Y, num_classes=num_class)
Val_X = change_to_new_channel(Val_X)
Val_Y = keras.utils.to_categorical(Val_Y_ori, num_classes=num_class)
Test_X = change_to_new_channel(Test_X)
Test_Y = keras.utils.to_categorical(Test_Y_ori, num_classes=num_class)
X_unlabeled = change_to_new_channel(X_unlabeled)


def min_max_scaler(input, min, max):
    """
    Min_max scaling of each channel.
    :param input:
    :param min:
    :param max:
    :return:
    """
    current_minmax = [(np.min(input[:, :, :, i]), np.max(input[:, :, :, i])) for i in range(new_channel)]
    for index, item in enumerate(current_minmax):
        input[:, :, :, index] = (input[:, :, :, index] - item[0])/(item[1] - item[0]) * (max - min) + min
    return input, current_minmax
# Min_max scaling
Train_X, current_minmax = min_max_scaler(Train_X, 0, 1)
for index, item in enumerate(current_minmax):
    Test_X[:, :, :, index] = (Test_X[:, :, :, index] - item[0]) / (item[1] - item[0])
    Val_X[:, :, :, index] = (Val_X[:, :, :, index] - item[0]) / (item[1] - item[0])
X_unlabeled, _ = min_max_scaler(X_unlabeled, 0, 1)

with open('paper2_data_for_DL_train_val_test.pickle', 'wb') as f:
    pickle.dump([Train_X, Train_Y, Val_X, Val_Y, Val_Y_ori, Test_X, Test_Y, Test_Y_ori, X_unlabeled], f)
