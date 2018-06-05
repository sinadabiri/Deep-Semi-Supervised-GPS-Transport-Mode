import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
# Settings
min_threshold = 20
max_threshold = 248
min_distance = 150  # Meters
min_time = 60  # Seconds
num_class = 5
new_channel = 4
min_percentile = 0
max_percentile = 100
# Import the final output from Instance_creation file, which is the filtered trips for all users.

filename = '../Mode-codes-Revised/paper2_trips_motion_features_NotFixedLength_woOutliers.pickle'
with open(filename, 'rb') as f:
    trip_motion_all_user_with_label, trip_motion_all_user_wo_label = pickle.load(f)
    #trip_motion_all_user_with_label = trip_motion_all_user_with_label[:1000]
    #trip_motion_all_user_wo_label = trip_motion_all_user_wo_label[:1000]


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

# Max_threshold=200: 200 is the rounded median size of all trips (i.e., GPS trajectory) after removing errors and
# outliers including: 1) max speed and acceleration, (2) trip length less than 10
X_labeled, Y_labeled_ori = trip_to_fixed_length(trip_motion_all_user_with_label, min_threshold=min_threshold,
                                                                max_threshold=max_threshold, min_distance=min_distance, min_time=min_time,
                                                                data_type='labeled')


def k_fold_stratified(X_labeled, Y_labeled_ori, fold=5):
    kfold_index = [[] for _ in range(fold)]
    for i in range(num_class):
        label_index = np.where(Y_labeled_ori == i)[0]
        for j in range(fold):
            portion = label_index[round(j*0.2*len(label_index)):round((j+1)*0.2*len(label_index))]
            kfold_index[j].append(portion)

    kfold_dataset = [[] for _ in range(num_class)]
    all_index = np.arange(0, len(Y_labeled_ori))
    for j in range(fold):
        test_index = np.hstack(tuple([label for label in kfold_index[j]]))
        Test_X = X_labeled[test_index]
        Test_Y_ori = Y_labeled_ori[test_index]
        train_index = np.delete(all_index, test_index)
        Train_X = X_labeled[train_index]
        Train_Y_ori = Y_labeled_ori[train_index]
        kfold_dataset[j] = [Train_X, Train_Y_ori, Test_X, Test_Y_ori]
    return kfold_dataset


def create_hand_crafted_features(X):
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
    for trip in X:
        RD = trip[0, :]
        DT = trip[1, :]
        VC = trip[2, :]
        SP = trip[3, :]
        AC = trip[4, :]
        J = trip[5, :]
        BR = trip[6, :]


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

    X_hand = [Dist, AV, EV, VV, MaxV1, MaxV2, MaxV3, MaxA1, MaxA2, MaxA3, HCR, SR, VCR]
    X_hand = np.array(X_hand, dtype=np.float32).T

    header = ['Distance', 'Average Velocity','Expectation Velocity', 'Variance of Velocity', 'MaxV1', 'MaxV2', 'MaxV3',
              'MaxA1', 'MaxA2', 'MaxA3', 'Heading Rate Change', 'Stop Rate', 'Velocity Change Rate', 'Label']
    return X_hand


def ml_fit_predict(ml_method, Train_X, Train_Y_ori, Test_X, Test_Y_ori):
    ml_method.fit(Train_X, Train_Y_ori)
    prediction = ml_method.predict(Test_X)
    test_acc = accuracy_score(Test_Y_ori, prediction)
    f1_macro = f1_score(Test_Y_ori, prediction, average='macro')
    f1_weight = f1_score(Test_Y_ori, prediction, average='weighted')
    return test_acc, f1_macro, f1_weight


def training_all_folds_ml(label_proportions, ml_method):
    kfold_dataset = k_fold_stratified(X_labeled, Y_labeled_ori, fold=5)
    test_accuracy_fold = [[] for _ in range(len(label_proportions))]
    mean_std_acc = [[] for _ in range(len(label_proportions))]
    test_metrics_fold = [[] for _ in range(len(label_proportions))]
    mean_std_metrics = [[] for _ in range(len(label_proportions))]
    for index, prop in enumerate(label_proportions):
        for i in range(len(kfold_dataset)):
            Train_X, Train_Y_ori, Test_X, Test_Y_ori = kfold_dataset[i]
            Test_X_hand = create_hand_crafted_features(Test_X)
            np.random.seed(7)
            random_sample = np.random.choice(len(Train_X), size=round(prop * len(Train_X)), replace=False, p=None)
            Train_X_sample = Train_X[random_sample]
            Train_Y_ori_sample = Train_Y_ori[random_sample]
            Train_X_hand = create_hand_crafted_features(Train_X_sample)
            test_acc, f1_macro, f1_weight = ml_fit_predict(ml_method, Train_X_hand, Train_Y_ori_sample, Test_X_hand, Test_Y_ori)
            test_accuracy_fold[index].append(test_acc)
            test_metrics_fold[index].append([f1_macro, f1_weight])

        accuracy_all = np.array(test_accuracy_fold[index])
        mean = np.mean(accuracy_all)
        std = np.std(accuracy_all)
        mean_std_acc[index] = [mean, std]
        metrics_all = np.array(test_metrics_fold[index])
        mean_metrics = np.mean(metrics_all, axis=0)
        std_metrics = np.std(metrics_all, axis=0)
        mean_std_metrics[index] = [mean_metrics, std_metrics]
    print(ml_method)
    for index, prop in enumerate(label_proportions):
        print('All Test Accuracy For ML Mehtod with Prop {} are: {}'.format(prop, test_accuracy_fold[index]))
        print('ML Method test accuracy for prop {}: Mean {}, std {}'.format(prop, mean_std_acc[index][0],
                                                                                  mean_std_acc[index][1]))
        print('ML Method test metrics for prop {}: Mean {}, std {}'.format(prop, mean_std_metrics[index][0],
                                                                                 mean_std_metrics[index][1]))
        print('\n')
        print('\n')

training_all_folds_ml(label_proportions=[0.1, 0.25, 0.50, 0.75, 1.0], ml_method=RandomForestClassifier())
training_all_folds_ml(label_proportions=[0.1, 0.25, 0.50, 0.75, 1.0], ml_method=KNeighborsClassifier())
training_all_folds_ml(label_proportions=[0.1, 0.25, 0.50, 0.75, 1.0], ml_method=SVC())
training_all_folds_ml(label_proportions=[0.1, 0.25, 0.50, 0.75, 1.0], ml_method=DecisionTreeClassifier())
training_all_folds_ml(label_proportions=[0.1, 0.25, 0.50, 0.75, 1.0], ml_method=MLPClassifier())
