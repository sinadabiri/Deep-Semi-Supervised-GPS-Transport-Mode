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
a = np.zeros([1, 2, 3, 4])
cs = os.getcwd()

# Settings
prop = 0.05  # the proportion of labeled data
batch_size = 100
latent_dim = 800
epochs_ae = 10
epochs_cls = 15
change = 10
units = 800  # num unit in the MLP hidden layer
num_filter = [32, 32, 64, 64, 128, 128]  # conv_layers and its channels for encoder
num_filter_cls = []  # conv_layers and its channels for cls
num_dense = 2  # number of dense layer in classifier excluding the last layer
kernel_size = (1, 3)
activation = tf.nn.relu
padding = 'same'
strides = 1
pool_size = (1, 2)

num_class = 5
reg_l2 = tf.contrib.layers.l1_regularizer(scale=0.1)
initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
#initializer = tf.truncated_normal_initializer()

#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
filename = '../Mode-codes-Revised/paper2_data_for_DL_train_val_test.pickle'
with open(filename, 'rb') as f:
    Train_X, Train_Y, Val_X, Val_Y, Val_Y_ori, Test_X, Test_Y, Test_Y_ori, X_unlabeled = pickle.load(f)
# Training and test set for GPS segments
random.seed(7)
np.random.seed(7)
tf.set_random_seed(7)
Train_X_Comb = np.vstack((X_unlabeled, Train_X))
Train_X_Unlabel = X_unlabeled
index = np.arange(len(Train_X))
np.random.shuffle(index)
Train_X = Train_X[index[:round(prop*len(Train_X))]]
Train_Y = Train_Y[index[:round(prop*len(Train_Y))]]
#Train_X_Comb = np.vstack((Train_X, Train_X_Unlabel))
random.shuffle(Train_X_Comb)

input_size = list(np.shape(Test_X)[1:])

# Encoder Network


def encoder_network(latent_dim, num_filter, input_combined, input_labeled):
    encoded_combined = input_combined
    encoded_labeled = input_labeled
    layers_shape = []
    for i in range(len(num_filter)):
        scope_name = 'encoder_set_' + str(i + 1)
        with tf.variable_scope(scope_name):
            encoded_combined = tf.layers.conv2d(inputs=encoded_combined, activation=tf.nn.relu, filters=num_filter[i],
                                                name='conv_1', kernel_size=kernel_size, strides=strides,
                                                padding=padding)
        with tf.variable_scope(scope_name, reuse=True):
            encoded_labeled = tf.layers.conv2d(inputs=encoded_labeled, activation=tf.nn.relu, filters=num_filter[i],
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
    latent_size = latent_combined.get_shape().as_list()

    return latent_combined, latent_labeled, layers_shape, latent_size

# # Decoder Network


def decoder_network(latent_combined, input_size, kernel_size, padding, activation):
    decoded_combined = latent_combined
    num_filter_ = num_filter[::-1]
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
                if i == len(num_filter_) - 1: # change it len(num_filter_)-1 if spatial size is not dividable by 2
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
                if i == len(num_filter_):  # change it len(num_filter_)-1 if spatial size is not dividable by 2
                    kernel_size = (1, input_size[1] - (decoded_combined.get_shape().as_list()[2] - 1) * strides)
                    padding = 'valid'
                decoded_combined = tf.layers.conv2d_transpose(inputs=decoded_combined, activation=activation,
                                                              filters=filter_size, name='deconv_1',
                                                              kernel_size=kernel_size,
                                                              strides=strides, padding=padding)
                if i != len(num_filter_) - 1:
                    decoded_combined = tf.keras.layers.UpSampling2D(name='UpSample', size=pool_size)(decoded_combined)

    return decoded_combined


def classifier_mlp(num_class, num_fliter_cls, num_dense, input_latent):
    conv_layer = input_latent
    for i in range(len(num_fliter_cls)):
        conv_layer = tf.layers.conv2d(inputs=conv_layer, activation=tf.nn.relu, filters=num_filter_cls[i],
                                  kernel_size=kernel_size, strides=strides, padding=padding,
                                  kernel_initializer=initializer)
        if len(num_filter) % 2 == 0:
            if i % 2 != 0:
                conv_layer = tf.layers.max_pooling2d(conv_layer, pool_size=pool_size,strides=pool_size, name='pool')
        else:
            if i % 2 == 0:
                conv_layer = tf.layers.max_pooling2d(conv_layer, pool_size=pool_size,strides=pool_size, name='pool')

    dense = tf.layers.flatten(conv_layer)
    scope_name = 'cls_set_'
    units = int(dense.get_shape().as_list()[-1] / 4)
    for i in range(num_dense):
        dense = tf.layers.dense(dense, units, activation=tf.nn.relu)
        units /= 2
    dense_last = dense
    dense = tf.layers.dropout(dense, 0.5)
    classifier_output = tf.layers.dense(dense, num_class, name='FC_4')
    return classifier_output, dense_last


def semi_supervised(input_latent, input_labeled, input_combined, true_label, alpha, beta, num_class, latent_dim, num_filter):

    decoded_output = decoder_network(latent_combined=latent_combined, input_size=input_size, kernel_size=kernel_size, padding=padding, activation=activation)
    classifier_output, dense = classifier_mlp(num_class=num_class, num_fliter_cls=num_filter_cls, num_dense=num_dense, input_latent=input_latent)

    loss_ae = tf.reduce_mean(tf.square(input_combined - decoded_output), name='loss_ae') * 100
    loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_label, logits=classifier_output),
                              name='loss_cls')
    total_loss = alpha*loss_ae + beta*loss_cls
    loss_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'EasyNet'))
    train_op_ae = tf.train.AdamOptimizer().minimize(loss_ae)
    train_op_cls = tf.train.AdamOptimizer().minimize(loss_cls)
    train_op = tf.train.AdamOptimizer().minimize(total_loss)
    # train_op = train_op = tf.layers.optimize_loss(total_loss, optimizer='Adam')

    correct_prediction = tf.equal(tf.argmax(true_label, 1), tf.argmax(classifier_output, 1))
    accuracy_cls = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return latent_size, loss_ae, loss_cls, accuracy_cls, train_op_ae, train_op_cls, classifier_output, dense, train_op, total_loss


def loss_acc_evaluation(Test_X, Test_Y, input):
    metrics = []
    for i in range(len(Test_X) // batch_size):
        Test_X_batch = Test_X[i * batch_size:(i + 1) * batch_size]
        Test_Y_batch = Test_Y[i * batch_size:(i + 1) * batch_size]
        loss_cls_, accuracy_cls_ = sess.run([loss_cls, accuracy_cls],
                                            feed_dict={input: Test_X_batch,
                                                       true_label: Test_Y_batch})
        metrics.append([loss_cls_, accuracy_cls_])
    Test_X_batch = Test_X[(i + 1) * batch_size:]
    Test_Y_batch = Test_Y[(i + 1) * batch_size:]
    if len(Test_X_batch) >= 1:
        loss_cls_, accuracy_cls_ = sess.run([loss_cls, accuracy_cls], feed_dict={input: Test_X_batch, true_label: Test_Y_batch})
        metrics.append([loss_cls_, accuracy_cls_])
    mean_ = np.mean(np.array(metrics), axis=0)
    print('Epoch Num {}, Loss_cls_Val {}, Accuracy_val {}'.format(k, mean_[0], mean_[1]))
    return mean_[0], mean_[1]

# Transfer the Train_X into latent layer of the trained AE.


def transfer_latent_space(X):
    latent_size = [len(X)] + latent_combined.get_shape().as_list()[1:]
    X_latent = np.zeros(latent_size)
    num_batches = len(X_latent) // batch_size
    for i in range(num_batches):
        X_latent[i*batch_size: (i + 1) * batch_size:] = sess.run(latent_combined,feed_dict={input_combined:X[i * batch_size: (i + 1) * batch_size]})
    X_latent[(i + 1) * batch_size:] = sess.run(latent_combined,
                                               feed_dict={input_combined: X[(i + 1) * batch_size:]})
    return X_latent


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


with tf.Session() as sess:
    input_labeled = tf.placeholder(dtype=tf.float32, shape=[None]+input_size, name='input_labeled')
    input_combined = tf.placeholder(dtype=tf.float32, shape=[None]+input_size, name='input_combined')
    true_label = tf.placeholder(tf.float32, shape=[None, num_class], name='true_label')
    alpha = tf.placeholder(tf.float32, shape=(), name='alpha')
    beta = tf.placeholder(tf.float32, shape=(), name='beta')
    latent_combined, latent_labeled, layers_shape, latent_size = encoder_network(latent_dim=latent_dim,
                                                                                 num_filter=num_filter,
                                                                                 input_combined=input_combined,
                                                                                 input_labeled=input_labeled)
    input_latent = tf.placeholder(dtype=tf.float32, shape=latent_size, name='input_labeled')
    latent_size, loss_ae, loss_cls, accuracy_cls, train_op_ae, train_op_cls, classifier_output, dense, train_op, total_loss = semi_supervised(input_latent=input_latent, input_labeled=input_labeled, input_combined=input_combined,
                                                                  true_label=true_label, alpha=alpha, beta=beta,
                                                                  num_class=num_class,
                                                                  latent_dim=latent_dim, num_filter=num_filter)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10)
    for k in range(epochs_ae):
        num_batches = len(Train_X_Comb) // batch_size
        x_combined_index = get_combined_index(train_x_comb=Train_X_Comb)
        for i in range(num_batches):
            unlab_index_range = x_combined_index[i * batch_size: (i + 1) * batch_size]
            X_ae = Train_X_Comb[unlab_index_range]
            loss_ae_, _ = sess.run([loss_ae, train_op_ae], feed_dict={input_combined: X_ae})
            print('Epoch Num {}, Batches Num {}, Loss_AE {}'.format
                  (k, i, np.round(loss_ae_, 3)))

        unlab_index_range = x_combined_index[(i + 1) * batch_size:]
        X_ae = Train_X_Comb[unlab_index_range]
        loss_ae_, _ = sess.run([loss_ae, train_op_ae], feed_dict={input_combined: X_ae})
        print('Epoch Num {}, Batches Num {}, Loss_AE {}'.format(k, i, np.round(loss_ae_, 3)))
    print('========================================================')
    print('End of training Autoencoder')
    print('========================================================')

    Val_X_latent = transfer_latent_space(Val_X)
    Test_X_latent = transfer_latent_space(Test_X)
    Train_X_latent = transfer_latent_space(Train_X)
    # Training classifier
    val_accuracy = {}
    val_loss = {}
    for k in range(epochs_cls):
        num_batches = len(Train_X_latent) // batch_size
        x_labeled_index = np.arange(len(Train_X_latent))
        for i in range(num_batches):
            lab_index_range = x_labeled_index[i * batch_size: (i + 1) * batch_size]
            X_cls = Train_X_latent[lab_index_range]
            Y_cls = Train_Y[lab_index_range]
            loss_cls_, accuracy_cls_, _ = sess.run([loss_cls, accuracy_cls, train_op_cls],
                                                   feed_dict={input_latent: X_cls, true_label: Y_cls})
            print('Epoch Num {}, Batches Num {}, Loss_cls {}, Accuracy_train {}'.format
                  (k, i, np.round(loss_cls_, 3), np.round(accuracy_cls_, 3)))

        lab_index_range = x_labeled_index[(i + 1) * batch_size:]
        X_cls = Train_X_latent[lab_index_range]
        Y_cls = Train_Y[lab_index_range]
        loss_cls_, accuracy_cls_, _ = sess.run([loss_cls, accuracy_cls, train_op_cls],
                                               feed_dict={input_latent: X_cls, true_label: Y_cls})
        print('Epoch Num {}, Batches Num {}, Loss_cls {}, Accuracy_train {}'.format
              (k, i, np.round(loss_cls_, 3), np.round(accuracy_cls_, 3)))

        print('====================================================')
        loss_val, acc_val = loss_acc_evaluation(Val_X_latent, Val_Y, input=input_latent)
        val_loss.update({k: loss_val})
        val_accuracy.update({k: acc_val})
        print('====================================================')
        saver.save(sess, "/Conv-Semi-TF-PS/" + str(prop), global_step=k)
    print("Val Accuracy Over Epochs: ", val_accuracy)
    print("Val loss Over Epochs: ", val_loss)

    max_acc = max(val_accuracy.items(), key=lambda k: k[1])[0]
    save_path = "/Conv-Semi-TF-PS/" + str(prop) + '-' + str(max_acc)
    saver.restore(sess, save_path)
    loss_Test, acc_Test = loss_acc_evaluation(Test_X_latent, Test_Y, input=input_latent)
    print('Conv-Semi-2steps test loss {} and test accuracy {}'.format(loss_Test, acc_Test))



