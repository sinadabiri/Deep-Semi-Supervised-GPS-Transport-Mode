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
# Settings
prop = 0.5  # the proportion of labeled data
batch_size = 100
latent_dim = 800
epochs_ae_cls = 20
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
filename = '../Mode-codes-Revised/paper2_data_for_DL_train_val_test.pickle'
with open(filename, 'rb') as f:
    Train_X, Train_Y, Val_X, Val_Y, Val_Y_ori, Test_X, Test_Y, Test_Y_ori, X_unlabeled = pickle.load(f)
# Training and test set for GPS segments
random.seed(7)
np.random.seed(7)
tf.set_random_seed(7)
Train_X_Comb = X_unlabeled
index = np.arange(len(Train_X))
np.random.shuffle(index)
Train_X = Train_X[index[:round(prop*len(Train_X))]]
Train_Y = Train_Y[index[:round(prop*len(Train_Y))]]
#Train_X_Comb = np.vstack((Train_X, Train_X_Unlabel))
random.shuffle(Train_X_Comb)

input_size = list(np.shape(Test_X)[1:])


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


def semi_supervised(input_labeled, input_combined, true_label, alpha, beta, num_class, latent_dim, num_filter_ae_cls, num_filter_cls, num_dense):

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


def loss_acc_evaluation(Test_X, Test_Y):
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
    print('Epoch Num {}, Loss_cls_Val {}, Accuracy_Val {}'.format(k, mean_[0], mean_[1]))
    return mean_[0], mean_[1]


def prediction_prob(Test_X):
    prediction = []
    for i in range(len(Test_X) // batch_size):
        Test_X_batch = Test_X[i * batch_size:(i + 1) * batch_size]
        prediction.append(sess.run(tf.nn.softmax(classifier_output), feed_dict={input_labeled: Test_X_batch}))
    Test_X_batch = Test_X[(i + 1) * batch_size:]
    prediction.append(sess.run(tf.nn.softmax(classifier_output), feed_dict={input_labeled: Test_X_batch}))
    prediction = np.vstack(tuple(prediction))
    return prediction

num_filter_ae_cls_all = [[32, 32], [32, 32, 64], [32, 32, 64, 64], [32, 32, 64, 64, 128],
                             [32, 32, 64, 64, 128, 128], [32, 32, 64, 64, 128, 128], [32, 32, 64, 64, 128, 128]]
num_filter_ae_cls_all = [[32, 32, 64, 64, 128, 128]]
class_posterior = []
orig_Train_X = Train_X.copy()
orig_Train_Y = Train_Y.copy()
for z in range(len(num_filter_ae_cls_all)):
    # Change the following see to None only for Ensemble.
    #random.seed(7)
    #np.random.seed(7)
    #tf.set_random_seed(7)
    tf.reset_default_graph()
    with tf.Session() as sess:
        input_labeled = tf.placeholder(dtype=tf.float32, shape=[None] + input_size, name='input_labeled')
        input_combined = tf.placeholder(dtype=tf.float32, shape=[None] + input_size, name='input_combined')
        true_label = tf.placeholder(tf.float32, shape=[None, num_class], name='true_label')
        alpha = tf.placeholder(tf.float32, shape=(), name='alpha')
        beta = tf.placeholder(tf.float32, shape=(), name='beta')

        num_filter_ae_cls = num_filter_ae_cls_all[z]
        loss_ae, loss_cls, accuracy_cls, train_op_ae, train_op_cls, classifier_output, dense, train_op, total_loss = semi_supervised(
            input_labeled=input_labeled, input_combined=input_combined, true_label=true_label, alpha=alpha, beta=beta,
            num_class=num_class, latent_dim=latent_dim, num_filter_ae_cls=num_filter_ae_cls,
            num_filter_cls=num_filter_cls, num_dense=num_dense)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=10)
        # Train_X, Train_Y = ensemble_train_set(orig_Train_X, orig_Train_Y)
        val_accuracy = {-2: 0, -1: 0}
        val_loss = {-2: 10, -1: 10}
        num_batches = len(Train_X_Comb) // batch_size
        #alfa_val1 = [0.0, 0.0, 1.0, 1.0, 1.0]
        #beta_val1 = [1.0, 1.0, 0.1, 0.1, 0.1]
        alfa_val = 0
        beta_val = 1
        change_to_ae = 1  # the value defines that algorithm is ready to change to joint ae-cls
        change_times = 0  # No. of times change from cls to ae-cls

        for k in range(epochs_ae_cls):
            # alfa_val = alfa_val1[k]
            # beta_val = beta_val1[k]

            x_combined_index = get_combined_index(train_x_comb=Train_X_Comb)
            x_labeled_index = get_labeled_index(train_x_comb=Train_X_Comb, train_x=Train_X)
            # x_labeled_index = np.arange(len(Train_X))
            for i in range(num_batches):
                unlab_index_range = x_combined_index[i * batch_size: (i + 1) * batch_size]
                lab_index_range = x_labeled_index[i * batch_size: (i + 1) * batch_size]
                X_ae = Train_X_Comb[unlab_index_range]
                X_cls = Train_X[lab_index_range]
                Y_cls = Train_Y[lab_index_range]
                loss_ae_, loss_cls_, accuracy_cls_, _ = sess.run([loss_ae, loss_cls, accuracy_cls, train_op],
                                                                 feed_dict={alpha: alfa_val, beta: beta_val,
                                                                            input_combined: X_ae,
                                                                            input_labeled: X_cls, true_label: Y_cls})
                print('Epoch Num {}, Batches Num {}, Loss_AE {}, Loss_cls {}, Accuracy_train {}'.format
                      (k, i, np.round(loss_ae_, 3), np.round(loss_cls_, 3), np.round(accuracy_cls_, 3)))

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
            loss_val, acc_val = loss_acc_evaluation(Val_X, Val_Y)
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
                alfa_val = 1
                beta_val += 0.1
                num_epoch_cls_only = k
                change_times += 1
                change_to_ae = 0
                key = 'change_' + str(k)
                val_accuracy.update({key: val_accuracy[k]})
                val_loss.update({key: val_loss[k]})
                val_accuracy.update({k: val_accuracy[max_acc] - 0.001})
                val_loss.update({k: val_loss[max_acc] + 0.001})

            elif all([not change_to_ae, val_accuracy[k] < val_accuracy[k - 1],
                      val_accuracy[k] < val_accuracy[k - 2]]):
                # save_path = "/Conv-Semi/" + str(prop) + '/' + str(k - 1) + ".ckpt"
                # #checkpoint = os.path.join(os.getcwd(), save_path)
                max_acc = max(val_accuracy.items(), key=lambda k: k[1])[0]
                #saver.restore(sess, "/Conv-Semi-TF-PS/" + str(prop) + '/' + str(max_acc) + ".ckpt")
                save_path = "/Conv-Semi-TF-PS/" + '2/' + str(z) + '/' + str(prop) + '-' + str(max_acc)
                saver.restore(sess, save_path)
                num_epoch_ae_cls = k - num_epoch_cls_only - 1
                alfa_val = 1
                beta_val = 0.2
                change_to_ae = 1
                key = 'change_' + str(k)
                val_accuracy.update({key: val_accuracy[k]})
                val_loss.update({key: val_loss[k]})
                val_accuracy.update({k: val_accuracy[max_acc] - 0.001})
                val_loss.update({k: val_loss[max_acc] + 0.001})
            if change_times == 2:
                break

        print("Ensembel {}: Val_Accu ae+cls Over Epochs {}: ".format(z, val_accuracy))
        print("Ensembel {}: Val_loss ae+cls Over Epochs {}: ".format(z, val_loss))
        # print("num_epoch_cls_only: ", num_epoch_cls_only)
        # print("num_epoch_ae_cls: ", num_epoch_ae_cls)
        class_posterior.append(prediction_prob(Test_X))

ave_class_posterior = sum(class_posterior)/len(class_posterior)
y_pred = np.argmax(ave_class_posterior, axis=1)
print('Test Accuracy of the Ensemble: ', accuracy_score(Test_Y_ori, y_pred))
a = 1

with tf.Session() as sess:

    # Pseudo label implementation mine
    ''''
    soft_max = tf.nn.softmax(classifier_output)
    pred_prob = []
    for i in range(len(Test_X) // batch_size):
        Test_X_batch = Test_X[i * batch_size:(i + 1) * batch_size]
        pred_prob.append(sess.run(soft_max, feed_dict={input_labeled: Test_X_batch}))
    Test_X_batch = Test_X[(i + 1) * batch_size:]
    pred_prob.append(sess.run(soft_max, feed_dict={input_labeled: Test_X_batch}))
    pred_prob = np.concatenate(pred_prob)
    y_pred = np.argmax(pred_prob, axis=1)
    index = np.where(y_pred == Test_Y_ori)[0]
    Accuracy_Test = len(index) * 1. / len(Test_Y_ori)
    pred_prob = pred_prob[index]
    y_pred = y_pred[index]
    percen_threshold = 0.8
    class_prob = [[], [], [], [], [], []]
    class_percentile = []
    for i in range(num_class):
        one_mode_pred_prob = np.max(pred_prob[np.where(y_pred == i)[0]], axis=1)
        class_prob[i] = one_mode_pred_prob
        class_percentile.append(np.percentile(one_mode_pred_prob, 80))
        print('Descriptive statistics for pred probability of class ', i,
              pd.Series(one_mode_pred_prob).describe(percentiles=list(np.linspace(0.05, 1, 11))))

    pred_prob_ul = []
    for i in range(len(Train_X_Comb) // batch_size):
        X_batch = Train_X_Comb[i * batch_size:(i + 1) * batch_size]
        pred_prob_ul.append(sess.run(soft_max, feed_dict={input_labeled: X_batch}))
    X_batch = Train_X_Comb[(i + 1) * batch_size:]
    pred_prob_ul.append(sess.run(soft_max, feed_dict={input_labeled: X_batch}))
    pred_prob_ul = np.concatenate(pred_prob_ul)
    Train_X_Unlabel = []
    Train_Y_Unlabel = []
    arg_max = np.argmax(pred_prob_ul, axis=1)
    for i, item in enumerate(arg_max):
        if np.max(pred_prob_ul[i]) >= class_percentile[item]:
            Train_X_Unlabel.append(Train_X_Comb[i])
            Train_Y_Unlabel.append(item)
    Train_X_Unlabel = np.array(Train_X_Unlabel)
    '''
    # Re-train the model
    '''
    sess.run(tf.global_variables_initializer())
    Train_X_Unlabel = np.vstack((Train_X_Unlabel, Train_X))
    Train_Y_Unlabel = sess.run(tf.one_hot(Train_Y_Unlabel, num_class))
    Train_Y_Unlabel = np.vstack((Train_Y_Unlabel, Train_Y))
    test_accuracy_pseudo = {}
    test_loss_pseudo = {}
    epochs = 30
    for k in range(epochs):
        for i in range(len(Train_X_Unlabel) // batch_size):
            X_cls = Train_X_Unlabel[i * batch_size: (i + 1) * batch_size]
            Y_cls = Train_Y_Unlabel[i * batch_size: (i + 1) * batch_size]
            loss_cls_, accuracy_cls_, _ = sess.run([loss_cls, accuracy_cls, train_op_cls],
                                                   feed_dict={input_labeled: X_cls, true_label: Y_cls})
            print('Epoch Num {}, Batches Num {}, Loss_cls {}, Accuracy_train {}'.format
                  (k, i, np.round(loss_cls_, 3), np.round(accuracy_cls_, 3)))

        X_cls = Train_X_Unlabel[(i + 1) * batch_size:]
        Y_cls = Train_Y_Unlabel[(i + 1) * batch_size:]
        loss_cls_, accuracy_cls_, _ = sess.run([loss_cls, accuracy_cls, train_op_cls],
                                               feed_dict={input_labeled: X_cls, true_label: Y_cls})
        print('Epoch Num {}, Batches Num {}, Loss_cls {}, Accuracy_train {}'.format
              (k, i, np.round(loss_cls_, 3), np.round(accuracy_cls_, 3)))
        print('==================================================================')
        metrics = []
        for i in range(len(Test_X) // batch_size):
            Test_X_batch = Test_X[i * batch_size:(i + 1) * batch_size]
            Test_Y_batch = Test_Y[i * batch_size:(i + 1) * batch_size]
            loss_cls_, accuracy_cls_ = sess.run([loss_cls, accuracy_cls],
                                            feed_dict={input_labeled: Test_X_batch, true_label: Test_Y_batch})
            metrics.append([loss_cls_, accuracy_cls_])
        Test_X_batch = Test_X[(i + 1) * batch_size:]
        Test_Y_batch = Test_Y[(i + 1) * batch_size:]
        loss_cls_, accuracy_cls_ = sess.run([loss_cls, accuracy_cls],
                                        feed_dict={input_labeled: Test_X_batch, true_label: Test_Y_batch})
        metrics.append([loss_cls_, accuracy_cls_])
        mean_ = np.mean(np.array(metrics), axis=0)
        print('Epoch Num {}, Loss_cls {}, Accuracy_test {}'.format(k, mean_[0], mean_[1]))
        print('==================================================================')
        test_loss_pseudo.update({k: mean_[0]})
        test_accuracy_pseudo.update({k: mean_[1]})

    print("Test Accuracy Over Epochs: ", test_accuracy_pseudo)
    print("Loss Classifier Over Epochs: ", test_loss_pseudo)
    '''


    # Implement classical ML
    '''
    x_labeled_index = np.arange(len(Train_X))
    X_latent = np.zeros((len(Train_X), dense.get_shape().as_list()[1]), dtype=np.float32)

    for i in range(len(Train_X) // batch_size):
        lab_index_range = x_labeled_index[i * batch_size: (i + 1) * batch_size]
        X_cls = Train_X[lab_index_range]
        Y_cls = Train_Y[lab_index_range]
        X_latent[i * batch_size: (i + 1) * batch_size] = sess.run(dense, feed_dict={input_labeled: X_cls})
    lab_index_range = x_labeled_index[(i + 1) * batch_size:]
    X_cls = Train_X[lab_index_range]
    X_latent[(i + 1) * batch_size:] = sess.run(dense, feed_dict={input_labeled: X_cls})
    Y_latent = sess.run(tf.argmax(true_label, axis=1), feed_dict={true_label: Train_Y})
    Test_X_latent = sess.run(dense, feed_dict={input_labeled: Test_X})
    # Apply PCA
    #pca = PCA(n_components=13)
    #X_latent = pca.fit_transform(X_latent)
    #Test_X_latent = pca.fit_transform(Test_X_latent)

    RandomForest = RandomForestClassifier()
    # parameters = {'C': [0.5, 1, 4, 7, 10, 13, 16, 20]}
    parameters = {'max_depth': [(i+1)*10 for i in range(10)]}
    clf = GridSearchCV(estimator=RandomForest, param_grid=parameters, cv=5)
    fit = clf.fit(X_latent, Y_latent)
    print('optimal parameter value: ', fit.best_params_)
    Prediction_RT = fit.best_estimator_.predict(Test_X_latent)
    #Accuracy_RandomForest = len(np.where(Prediction_RT == y_test)[0]) * 1. / len(y_test)
    #print('Accuracy: ', Accuracy_RandomForest)
    print('Test Accuracy Random Forest%: ', accuracy_score(Test_Y_ori, Prediction_RT))
    #print(classification_report(y_test, Prediction_RT, digits=3))

    # SVM
    RandomForest = SVC()
    parameters = {'C': [0.5, 1, 4, 7, 10, 13, 16, 20]}
    clf = GridSearchCV(estimator=RandomForest, param_grid=parameters, cv=5)
    fit = clf.fit(X_latent, Y_latent)
    print('optimal parameter value: ', fit.best_params_)
    Prediction_RT = fit.best_estimator_.predict(Test_X_latent)
    # Accuracy_RandomForest = len(np.where(Prediction_RT == y_test)[0]) * 1. / len(y_test)
    # print('Accuracy: ', Accuracy_RandomForest)
    print('Test Accuracy SVC %: ', accuracy_score(Test_Y_ori, Prediction_RT))
    # print(classification_report(y_test, Prediction_RT, digits=3))
    #===================

    y_pred = sess.run(tf.argmax(input=classifier_output, axis=1), feed_dict={input_labeled: Test_X})
    print('Test Accuracy %: ', accuracy_score(Test_Y_ori, y_pred))
    print('\n')
    print('Confusin matrix: ', confusion_matrix(Test_Y_ori, y_pred))
    print('\n')
    print(classification_report(Test_Y_ori, y_pred, digits=3))
    '''


