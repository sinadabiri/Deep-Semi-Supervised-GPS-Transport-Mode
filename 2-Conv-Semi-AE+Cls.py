import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import random
import pickle
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import keras

y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
print(confusion_matrix(y_true, y_pred))

# Training Settings
batch_size = 100
latent_dim = 800
change = 10
units = 800  # num unit in the MLP hidden layer
num_filter_ae_cls = [32, 32, 64, 64, 128, 128]  # conv_layers and No. of its channels for AE + CLS
num_filter_cls = []  # conv_layers and No. of its channel for only cls
num_dense = 0  # number of dense layer in classifier excluding the last layer
kernel_size = (1, 3)
activation = tf.nn.relu
padding = 'same'
strides = 1
pool_size = (1, 2)
num_class = 5
reg_l2 = tf.contrib.layers.l1_regularizer(scale=0.1)
initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
#initializer = tf.truncated_normal_initializer()

# Import the data
#filename = '../Mode-codes-Revised/paper2_data_for_DL_train_val_test.pickle'
filename = '../Mode-codes-Revised/paper2_data_for_DL_kfold_dataset_RL.pickle'
with open(filename, 'rb') as f:
    kfold_dataset, X_unlabeled = pickle.load(f)


# Encoder Network


def encoder_network(latent_dim, num_filter_ae_cls, input_combined, input_labeled):
    encoded_combined = input_combined
    encoded_labeled = input_labeled
    layers_shape = []
    for i in range(len(num_filter_ae_cls)):
        scope_name = 'encoder_set_' + str(i + 1)
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE, initializer=initializer):
            encoded_combined = tf.layers.conv2d(inputs=encoded_combined, activation=tf.nn.relu, filters=num_filter_ae_cls[i],
                                                name='conv_1', kernel_size=kernel_size, strides=strides,
                                                padding=padding)
        with tf.variable_scope(scope_name, reuse=True, initializer=initializer):
            encoded_labeled = tf.layers.conv2d(inputs=encoded_labeled, activation=tf.nn.relu, filters=num_filter_ae_cls[i],
                                               name='conv_1', kernel_size=kernel_size, strides=strides, padding=padding)

        if i % 2 != 0:
            encoded_combined = tf.layers.max_pooling2d(encoded_combined, pool_size=pool_size,
                                                          strides=pool_size, name='pool')
            encoded_labeled = tf.layers.max_pooling2d(encoded_labeled, pool_size=pool_size,
                                                          strides=pool_size, name='pool')
        layers_shape.append(encoded_combined.get_shape().as_list())

    layers_shape.append(encoded_combined.get_shape().as_list())
    latent_combined = encoded_combined
    latent_labeled = encoded_labeled

    return latent_combined, latent_labeled, layers_shape

# # Decoder Network


def decoder_network(latent_combined, input_size, kernel_size, padding, activation):
    decoded_combined = latent_combined
    num_filter_ = num_filter_ae_cls[::-1]
    if len(num_filter_) % 2 == 0:
        num_filter_ = sorted(set(num_filter_), reverse=True)
        for i in range(len(num_filter_)):
            decoded_combined = tf.keras.layers.UpSampling2D(name='UpSample', size=pool_size)(decoded_combined)
            scope_name = 'decoder_set_' + str(2*i)
            with tf.variable_scope(scope_name, initializer=initializer):
                decoded_combined = tf.layers.conv2d_transpose(inputs=decoded_combined, activation=activation,
                                                              filters=num_filter_[i], name='deconv_1',
                                                              kernel_size=kernel_size,
                                                              strides=strides, padding=padding)
            scope_name = 'decoder_set_' + str(2*i + 1)
            with tf.variable_scope(scope_name, initializer=initializer):
                filter_size, activation = (input_size[-1], tf.nn.sigmoid) if i == len(num_filter_) - 1 else (int(num_filter_[i] / 2), tf.nn.relu)
                if i == len(num_filter_): # change it len(num_filter_)-1 if spatial size is not dividable by 2
                    kernel_size = (1, input_size[1] - (decoded_combined.get_shape().as_list()[2] - 1) * strides)
                    padding = 'valid'
                decoded_combined = tf.layers.conv2d_transpose(inputs=decoded_combined, activation=activation,
                                                              filters=filter_size, name='deconv_1',
                                                              kernel_size=kernel_size,
                                                              strides=strides, padding=padding)
    else:
        num_filter_ = sorted(set(num_filter_), reverse=True)
        for i in range(len(num_filter_)):
            scope_name = 'decoder_set_' + str(2 * i)
            with tf.variable_scope(scope_name, initializer=initializer):
                decoded_combined = tf.layers.conv2d_transpose(inputs=decoded_combined, activation=activation,
                                                              filters=num_filter_[i], name='deconv_1',
                                                              kernel_size=kernel_size,
                                                              strides=strides, padding=padding)
            scope_name = 'decoder_set_' + str(2 * i + 1)
            with tf.variable_scope(scope_name, initializer=initializer):
                filter_size, activation = (input_size[-1], tf.nn.sigmoid) if i == len(num_filter_) - 1 else (int(num_filter_[i] / 2), tf.nn.relu)
                if i == len(num_filter_): # change it len(num_filter_)-1 if spatial size is not dividable by 2
                    kernel_size = (1, input_size[1] - (decoded_combined.get_shape().as_list()[2] - 1) * strides)
                    padding = 'valid'
                decoded_combined = tf.layers.conv2d_transpose(inputs=decoded_combined, activation=activation,
                                                              filters=filter_size, name='deconv_1',
                                                              kernel_size=kernel_size,
                                                              strides=strides, padding=padding)
                if i != len(num_filter_) - 1:
                    decoded_combined = tf.keras.layers.UpSampling2D(name='UpSample', size=pool_size)(decoded_combined)

    return decoded_combined


def classifier_mlp(latent_labeled, num_class, num_filter_cls, num_dense):
    conv_layer = latent_labeled
    for i in range(len(num_filter_cls)):
        scope_name = 'cls_conv_set_' + str(i + 1)
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE, initializer=initializer):
            conv_layer = tf.layers.conv2d(inputs=conv_layer, activation=tf.nn.relu, filters=num_filter_cls[i],
                                          kernel_size=kernel_size, strides=strides, padding=padding,
                                          kernel_initializer=initializer)
        if len(num_filter_cls) % 2 == 0:
            if i % 2 != 0:
                conv_layer = tf.layers.max_pooling2d(conv_layer, pool_size=pool_size,strides=pool_size, name='pool')
        else:
            if i % 2 == 0:
                conv_layer = tf.layers.max_pooling2d(conv_layer, pool_size=pool_size,strides=pool_size, name='pool')

    dense = tf.layers.flatten(conv_layer)
    units = int(dense.get_shape().as_list()[-1] / 4)
    for i in range(num_dense):
        scope_name = 'cls_dense_set_' + str(i + 1)
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE, initializer=initializer):
            dense = tf.layers.dense(dense, units, activation=tf.nn.relu, kernel_initializer=initializer)
        units /= 2
    dense_last = dense
    dense = tf.layers.dropout(dense, 0.5)
    scope_name = 'cls_last_dense_'
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE, initializer=initializer):
        classifier_output = tf.layers.dense(dense, num_class, name='FC_4', kernel_initializer=initializer)
    return classifier_output, dense_last


def semi_supervised(input_labeled, input_combined, true_label, alpha, beta, num_class, latent_dim, num_filter_ae_cls, num_filter_cls, num_dense, input_size):

    latent_combined, latent_labeled, layers_shape = encoder_network(latent_dim=latent_dim, num_filter_ae_cls=num_filter_ae_cls,
                                                                    input_combined=input_combined, input_labeled=input_labeled)
    decoded_output = decoder_network(latent_combined=latent_combined, input_size=input_size, kernel_size=kernel_size, activation=activation, padding=padding)
    classifier_output, dense = classifier_mlp(latent_labeled, num_class, num_filter_cls=num_filter_cls, num_dense=num_dense)
    #classifier_output = classifier_cnn(latent_labeled, num_filter=num_filter)

    loss_ae = tf.reduce_mean(tf.square(input_combined - decoded_output), name='loss_ae') * 100
    loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_label, logits=classifier_output),
                              name='loss_cls')
    total_loss = alpha*loss_ae + beta*loss_cls
    #total_loss = beta * loss_ae + alpha * loss_cls
    loss_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'EasyNet'))
    train_op_ae = tf.train.AdamOptimizer().minimize(loss_ae)
    train_op_cls = tf.train.AdamOptimizer().minimize(loss_cls)
    train_op = tf.train.AdamOptimizer().minimize(total_loss)
    # train_op = train_op = tf.layers.optimize_loss(total_loss, optimizer='Adam')

    correct_prediction = tf.equal(tf.argmax(true_label, 1), tf.argmax(classifier_output, 1))
    accuracy_cls = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return loss_ae, loss_cls, accuracy_cls, train_op_ae, train_op_cls, classifier_output, dense, train_op, total_loss


def get_combined_index(train_x_comb):
    x_combined_index = np.arange(len(train_x_comb))
    np.random.shuffle(x_combined_index)
    return x_combined_index


def get_labeled_index(train_x_comb, train_x):
    labeled_index = []
    for i in range(len(train_x_comb) // len(train_x)):
        l = np.arange(len(train_x))
        np.random.shuffle(l)
        labeled_index.append(l)
    labeled_index.append(np.arange(len(train_x_comb) % len(train_x)))
    return np.concatenate(labeled_index)


def ensemble_train_set(Train_X, Train_Y):
    index = np.random.choice(len(Train_X), size=len(Train_X), replace=True, p=None)
    return Train_X[index], Train_Y[index]


def loss_acc_evaluation(Test_X, Test_Y, loss_cls, accuracy_cls, input_labeled, true_label, k, sess):
    metrics = []
    for i in range(len(Test_X) // batch_size):
        Test_X_batch = Test_X[i * batch_size:(i + 1) * batch_size]
        Test_Y_batch = Test_Y[i * batch_size:(i + 1) * batch_size]
        loss_cls_, accuracy_cls_ = sess.run([loss_cls, accuracy_cls],
                                            feed_dict={input_labeled: Test_X_batch,
                                                       true_label: Test_Y_batch})
        metrics.append([loss_cls_, accuracy_cls_])
    Test_X_batch = Test_X[(i + 1) * batch_size:]
    Test_Y_batch = Test_Y[(i + 1) * batch_size:]
    if len(Test_X_batch) >= 1:
        loss_cls_, accuracy_cls_ = sess.run([loss_cls, accuracy_cls],
                                        feed_dict={input_labeled: Test_X_batch,
                                                   true_label: Test_Y_batch})
        metrics.append([loss_cls_, accuracy_cls_])
    mean_ = np.mean(np.array(metrics), axis=0)
    #print('Epoch Num {}, Loss_cls_Val {}, Accuracy_Val {}'.format(k, mean_[0], mean_[1]))
    return mean_[0], mean_[1]


def prediction_prob(Test_X, classifier_output, input_labeled, sess):
    prediction = []
    for i in range(len(Test_X) // batch_size):
        Test_X_batch = Test_X[i * batch_size:(i + 1) * batch_size]
        prediction.append(sess.run(tf.nn.softmax(classifier_output), feed_dict={input_labeled: Test_X_batch}))
    Test_X_batch = Test_X[(i + 1) * batch_size:]
    prediction.append(sess.run(tf.nn.softmax(classifier_output), feed_dict={input_labeled: Test_X_batch}))
    prediction = np.vstack(tuple(prediction))
    return prediction


def train_val_split(Train_X, Train_Y_ori):
    val_index = []
    for i in range(num_class):
        label_index = np.where(Train_Y_ori == i)[0]
        val_index.append(label_index[:round(0.1*len(label_index))])
    val_index = np.hstack(tuple([label for label in val_index]))
    Val_X = Train_X[val_index]
    Val_Y_ori = Train_Y_ori[val_index]
    Val_Y = keras.utils.to_categorical(Val_Y_ori, num_classes=num_class)
    train_index_ = np.delete(np.arange(0, len(Train_Y_ori)), val_index)
    Train_X = Train_X[train_index_]
    Train_Y_ori = Train_Y_ori[train_index_]
    Train_Y = keras.utils.to_categorical(Train_Y_ori, num_classes=num_class)
    return Train_X, Train_Y, Train_Y_ori, Val_X, Val_Y, Val_Y_ori


def training(one_fold, X_unlabeled, seed, prop, num_filter_ae_cls_all, epochs_ae_cls=20):
    Train_X = one_fold[0]
    Train_Y_ori = one_fold[1]
    random.seed(seed)
    np.random.seed(seed)
    random_sample = np.random.choice(len(Train_X), size=round(0.5*len(Train_X)), replace=False, p=None)
    Train_X = Train_X[random_sample]
    Train_Y_ori = Train_Y_ori[random_sample]
    Train_X, Train_Y, Train_Y_ori, Val_X, Val_Y, Val_Y_ori = train_val_split(Train_X, Train_Y_ori)
    Test_X = one_fold[2]
    Test_Y = one_fold[3]
    Test_Y_ori = one_fold[4]
    random_sample = np.random.choice(len(X_unlabeled), size=round(prop * len(X_unlabeled)), replace=False, p=None)
    X_unlabeled = X_unlabeled[random_sample]
    Train_X_Comb = X_unlabeled

    input_size = list(np.shape(Test_X)[1:])
    # Various sets of number of filters for ensemble. If choose one set, no ensemble is implemented.
    num_filter_ae_cls_all = [[32, 32], [32, 32, 64], [32, 32, 64, 64], [32, 32, 64, 64, 128],
                             [32, 32, 64, 64, 128, 128], [32, 32, 64, 64, 128, 128], [32, 32, 64, 64, 128, 128]]
    num_filter_ae_cls_all = [[32, 32, 64, 64, 128, 128]]
    class_posterior = []

    # This for loop is only for implementing ensemble
    for z in range(len(num_filter_ae_cls_all)):
        # Change the following seed to None only for Ensemble.
        tf.reset_default_graph()  # Used for ensemble
        with tf.Session() as sess:
            input_labeled = tf.placeholder(dtype=tf.float32, shape=[None] + input_size, name='input_labeled')
            input_combined = tf.placeholder(dtype=tf.float32, shape=[None] + input_size, name='input_combined')
            true_label = tf.placeholder(tf.float32, shape=[None, num_class], name='true_label')
            alpha = tf.placeholder(tf.float32, shape=(), name='alpha')
            beta = tf.placeholder(tf.float32, shape=(), name='beta')

            num_filter_ae_cls = num_filter_ae_cls_all[z]
            loss_ae, loss_cls, accuracy_cls, train_op_ae, train_op_cls, classifier_output, dense, train_op, total_loss = semi_supervised(
                input_labeled=input_labeled, input_combined=input_combined, true_label=true_label, alpha=alpha,
                beta=beta, num_class=num_class, latent_dim=latent_dim, num_filter_ae_cls=num_filter_ae_cls,
                num_filter_cls=num_filter_cls, num_dense=num_dense, input_size=input_size)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=20)
            # Train_X, Train_Y = ensemble_train_set(orig_Train_X, orig_Train_Y)
            val_accuracy = {-2: 0, -1: 0}
            val_loss = {-2: 10, -1: 10}
            num_batches = len(Train_X_Comb) // batch_size
            # alfa_val1 = [0.0, 0.0, 1.0, 1.0, 1.0]
            # beta_val1 = [1.0, 1.0, 0.1, 0.1, 0.1]
            alfa_val = 1  ## 0
            beta_val = 1
            change_to_ae = 1  # the value defines that algorithm is ready to change to joint ae-cls
            change_times = 0  # No. of times change from cls to ae-cls, which is 2 for this training strategy
            third_step = 0
            for k in range(epochs_ae_cls):
                # alfa_val = alfa_val1[k]
                # beta_val = beta_val1[k]

                #beta_val = min(((1 - 0.1) / (-epochs_ae_cls)) * k + 1, 0.1) ##
                #alfa_val = max(((1.5 - 1) / (epochs_ae_cls)) * k + 1, 1.5)

                x_combined_index = get_combined_index(train_x_comb=Train_X_Comb)
                x_labeled_index = get_labeled_index(train_x_comb=Train_X_Comb, train_x=Train_X)
                for i in range(num_batches):
                    unlab_index_range = x_combined_index[i * batch_size: (i + 1) * batch_size]
                    lab_index_range = x_labeled_index[i * batch_size: (i + 1) * batch_size]
                    X_ae = Train_X_Comb[unlab_index_range]
                    X_cls = Train_X[lab_index_range]
                    Y_cls = Train_Y[lab_index_range]
                    loss_ae_, loss_cls_, accuracy_cls_, _ = sess.run([loss_ae, loss_cls, accuracy_cls, train_op],
                                                                     feed_dict={alpha: alfa_val, beta: beta_val,
                                                                                input_combined: X_ae,
                                                                                input_labeled: X_cls,
                                                                                true_label: Y_cls})
                    #print('Epoch Num {}, Batches Num {}, Loss_AE {}, Loss_cls {}, Accuracy_train {}'.format
                          #(k, i, np.round(loss_ae_, 3), np.round(loss_cls_, 3), np.round(accuracy_cls_, 3)))

                unlab_index_range = x_combined_index[(i + 1) * batch_size:]
                lab_index_range = x_labeled_index[(i + 1) * batch_size:]
                X_ae = Train_X_Comb[unlab_index_range]
                X_cls = Train_X[lab_index_range]
                Y_cls = Train_Y[lab_index_range]
                loss_ae_, loss_cls_, accuracy_cls_, _ = sess.run([loss_ae, loss_cls, accuracy_cls, train_op],
                                                                 feed_dict={alpha: alfa_val, beta: beta_val,
                                                                            input_combined: X_ae,
                                                                            input_labeled: X_cls, true_label: Y_cls})
                print('Epoch Num {}, Batches Num {}, Loss_AE {}, Loss_cls {}, Accuracy_train {}'.format
                      (k, i, np.round(loss_ae_, 3), np.round(loss_cls_, 3), np.round(accuracy_cls_, 3)))

                print('====================================================')
                loss_val, acc_val = loss_acc_evaluation(Val_X, Val_Y, loss_cls, accuracy_cls, input_labeled, true_label, k, sess)
                val_loss.update({k: loss_val})
                val_accuracy.update({k: acc_val})
                print('====================================================')
                saver.save(sess, "/Conv-Semi-TF-PS/" + '2/' + str(z) + '/' + str(prop), global_step=k)
                # save_path = "/Conv-Semi/" + str(prop) + '/' + str(k) + ".ckpt"
                # checkpoint = os.path.join(os.getcwd(), save_path)
                # saver.save(sess, checkpoint)
                # if alfa_val == 1:
                # beta_val += 0.05

                if all([change_to_ae, val_accuracy[k] < val_accuracy[k - 1], val_accuracy[k] < val_accuracy[k - 2]]):
                    # save_path = "/Conv-Semi/" + str(prop) + '/' + str(k-1) + ".ckpt"
                    # checkpoint = os.path.join(os.getcwd(), save_path)
                    max_acc = max(val_accuracy.items(), key=lambda k: k[1])[0]
                    save_path = "/Conv-Semi-TF-PS/" + '2/' + str(z) + '/' + str(prop) + '-' + str(max_acc)
                    saver.restore(sess, save_path)
                    alfa_val = 1.0
                    beta_val = 0.1
                    num_epoch_cls_only = k
                    change_times += 1
                    change_to_ae = 1
                    key = 'change_' + str(k)
                    val_accuracy.update({key: val_accuracy[k]})
                    val_loss.update({key: val_loss[k]})
                    #val_accuracy.update({k: val_accuracy[max_acc] - 0.001}) ##
                    #val_loss.update({k: val_loss[max_acc] + 0.001})  ##

                elif all([not change_to_ae, val_accuracy[k] < val_accuracy[k - 1],
                          val_accuracy[k] < val_accuracy[k - 2]]):
                    # save_path = "/Conv-Semi/" + str(prop) + '/' + str(k - 1) + ".ckpt"
                    # #checkpoint = os.path.join(os.getcwd(), save_path)
                    max_acc = max(val_accuracy.items(), key=lambda k: k[1])[0]
                    # saver.restore(sess, "/Conv-Semi-TF-PS/" + str(prop) + '/' + str(max_acc) + ".ckpt")
                    save_path = "/Conv-Semi-TF-PS/" + '2/' + str(z) + '/' + str(prop) + '-' + str(max_acc)
                    saver.restore(sess, save_path)
                    num_epoch_ae_cls = k - num_epoch_cls_only - 1
                    alfa_val = 1.5
                    beta_val = 0.2
                    change_times += 1  ##
                    change_to_ae = 1
                    key = 'change_' + str(k)
                    val_accuracy.update({key: val_accuracy[k]})
                    val_loss.update({key: val_loss[k]})
                    #val_accuracy.update({k: val_accuracy[max_acc] - 0.001})  ##
                    #val_loss.update({k: val_loss[max_acc] + 0.001})  ##
                if change_times == 2: ##
                    break

            print("Ensemble {}: Val_Accu ae+cls Over Epochs {}: ".format(z, val_accuracy))
            print("Ensemble {}: Val_loss ae+cls Over Epochs {}: ".format(z, val_loss))
            class_posterior.append(prediction_prob(Test_X, classifier_output, input_labeled, sess))

        ave_class_posterior = sum(class_posterior) / len(class_posterior)
        y_pred = np.argmax(ave_class_posterior, axis=1)
        test_accuracy = accuracy_score(Test_Y_ori, y_pred)
        #precision = precision_score(Test_Y_ori, y_pred, average='weighted')
        #recall = recall_score(Test_Y_ori, y_pred, average='weighted')
        f1_macro = f1_score(Test_Y_ori, y_pred, average='macro')
        f1_weight = f1_score(Test_Y_ori, y_pred, average='weighted')
        print('Semi-AE+Cls Test Accuracy of the Ensemble: ', test_accuracy)
        print('Confusion Matrix: ', confusion_matrix(Test_Y_ori, y_pred))

    return test_accuracy, f1_macro, f1_weight

def training_all_folds(label_proportions, num_filter):
    test_accuracy_fold = [[] for _ in range(len(label_proportions))]
    mean_std_acc = [[] for _ in range(len(label_proportions))]
    test_metrics_fold = [[] for _ in range(len(label_proportions))]
    mean_std_metrics = [[] for _ in range(len(label_proportions))]
    for index, prop in enumerate(label_proportions):
        for i in range(len(kfold_dataset)):
            test_accuracy, f1_macro, f1_weight = training(kfold_dataset[i], X_unlabeled=X_unlabeled, seed=7, prop=prop, num_filter_ae_cls_all=num_filter)
            test_accuracy_fold[index].append(test_accuracy)
            test_metrics_fold[index].append([f1_macro, f1_weight])
        accuracy_all = np.array(test_accuracy_fold[index])
        mean = np.mean(accuracy_all)
        std = np.std(accuracy_all)
        mean_std_acc[index] = [mean, std]
        metrics_all = np.array(test_metrics_fold[index])
        mean_metrics = np.mean(metrics_all, axis=0)
        std_metrics = np.std(metrics_all, axis=0)
        mean_std_metrics[index] = [mean_metrics, std_metrics]
    for index, prop in enumerate(label_proportions):
        print('All Test Accuracy For Semi-AE+Cls with Prop {} are: {}'.format(prop, test_accuracy_fold[index]))
        print('Semi-AE+Cls test accuracy for prop {}: Mean {}, std {}'.format(prop, mean_std_acc[index][0], mean_std_acc[index][1]))
        print('Semi-AE+Cls test metrics for prop {}: Mean {}, std {}'.format(prop, mean_std_metrics[index][0], mean_std_metrics[index][1]))
        print('\n')
    return test_accuracy_fold, test_metrics_fold, mean_std_acc, mean_std_metrics

test_accuracy_fold, test_metrics_fold, mean_std_acc, mean_std_metrics = training_all_folds(
    label_proportions=[0.15, 0.35], num_filter=[32, 32, 64, 64])


