import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import random
import pickle
import tensorflow as tf
import keras
import os
from sklearn.decomposition import PCA

mean_std = [[] for _ in range(5)]
mean_std[0] = [1, 2]
b = 1


# Settings
batch_size = 100
latent_dim = 800
change = 10
units = 800  # num unit in the MLP hidden layer
num_filter_cls = []  # conv_layers and its channels for cls
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

#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
filename = '../Mode-codes-Revised/paper2_data_for_DL_kfold_dataset.pickle'
with open(filename, 'rb') as f:
    kfold_dataset, X_unlabeled = pickle.load(f)

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


def decoder_network(latent_combined, input_size, kernel_size, padding, activation, num_filter):
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


def classifier_mlp(num_class, num_fliter_cls, num_dense, input_latent, num_filter):
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


def semi_supervised(input_latent, input_labeled, input_combined, true_label, alpha, beta, num_class, latent_dim, num_filter, latent_combined, input_size):

    decoded_output = decoder_network(latent_combined=latent_combined, input_size=input_size, kernel_size=kernel_size, padding=padding, activation=activation, num_filter=num_filter)
    classifier_output, dense = classifier_mlp(num_class=num_class, num_fliter_cls=num_filter_cls, num_dense=num_dense, input_latent=input_latent, num_filter=num_filter)

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
    return loss_ae, loss_cls, accuracy_cls, train_op_ae, train_op_cls, classifier_output, dense, train_op, total_loss


def loss_acc_evaluation(Test_X, Test_Y, input, sess, loss_cls, accuracy_cls, true_label, k):
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


def prediction_prob(Test_X, classifier_output, input_labeled, sess):
    prediction = []
    for i in range(len(Test_X) // batch_size):
        Test_X_batch = Test_X[i * batch_size:(i + 1) * batch_size]
        prediction.append(sess.run(tf.nn.softmax(classifier_output), feed_dict={input_labeled: Test_X_batch}))
    Test_X_batch = Test_X[(i + 1) * batch_size:]
    if len(Test_X_batch) >= 1:
        prediction.append(sess.run(tf.nn.softmax(classifier_output), feed_dict={input_labeled: Test_X_batch}))
    prediction = np.vstack(tuple(prediction))
    y_pred = np.argmax(prediction, axis=1)
    return y_pred


def transfer_latent_space(X, latent_combined, sess, input_combined):
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

def rand_stra_sample(Train_X, Train_Y_ori, prop):
    sampled_index = []
    for i in range(num_class):
        index = np.where(Train_Y_ori == i)[0]
        sampled_index.append(np.random.choice(index, size=round(prop*len(index)), replace=False, p=None))
    sampled_index = np.hstack(tuple(sampled_index))
    Train_X = Train_X[sampled_index]
    Train_Y_ori = Train_Y_ori[sampled_index]
    return Train_X, Train_Y_ori


def training(one_fold, X_unlabeled, seed, prop, num_filter, epochs_ae=10, epochs_cls=20):
    Train_X = one_fold[0]
    Train_Y_ori = one_fold[1]
    Test_X = one_fold[2]
    Test_Y = one_fold[3]
    Test_Y_ori = one_fold[4]
    random.seed(seed)
    np.random.seed(seed)
    random_sample = np.random.choice(len(Train_X), size=round(0.5*len(Train_X)), replace=False, p=None)
    Train_X = Train_X[random_sample]
    Train_Y_ori = Train_Y_ori[random_sample]
    #Train_X, Train_Y_ori = rand_stra_sample(Train_X, Train_Y_ori, prop)
    Train_X, Train_Y, Train_Y_ori, Val_X, Val_Y, Val_Y_ori = train_val_split(Train_X, Train_Y_ori)
    random_sample = np.random.choice(len(X_unlabeled), size=round(prop * len(X_unlabeled)), replace=False, p=None)
    X_unlabeled = X_unlabeled[random_sample]
    Train_X_Comb = np.vstack((X_unlabeled, Train_X))
    np.random.shuffle(Train_X_Comb)
    input_size = list(np.shape(Test_X)[1:])

    tf.reset_default_graph()
    with tf.Session() as sess:
        input_labeled = tf.placeholder(dtype=tf.float32, shape=[None] + input_size, name='input_labeled')
        input_combined = tf.placeholder(dtype=tf.float32, shape=[None] + input_size, name='input_combined')
        true_label = tf.placeholder(tf.float32, shape=[None, num_class], name='true_label')
        alpha = tf.placeholder(tf.float32, shape=(), name='alpha')
        beta = tf.placeholder(tf.float32, shape=(), name='beta')
        latent_combined, latent_labeled, layers_shape, latent_size = encoder_network(latent_dim=latent_dim,
                                                                                     num_filter=num_filter,
                                                                                     input_combined=input_combined,
                                                                                     input_labeled=input_labeled)
        input_latent = tf.placeholder(dtype=tf.float32, shape=latent_size, name='input_labeled')
        loss_ae, loss_cls, accuracy_cls, train_op_ae, train_op_cls, classifier_output, dense, train_op, total_loss = semi_supervised(
            input_latent=input_latent, input_labeled=input_labeled, input_combined=input_combined,
            true_label=true_label, alpha=alpha, beta=beta,
            num_class=num_class,
            latent_dim=latent_dim, num_filter=num_filter, latent_combined=latent_combined, input_size=input_size)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=20)
        for k in range(epochs_ae):
            num_batches = len(Train_X_Comb) // batch_size
            x_combined_index = get_combined_index(train_x_comb=Train_X_Comb)
            for i in range(num_batches):
                unlab_index_range = x_combined_index[i * batch_size: (i + 1) * batch_size]
                X_ae = Train_X_Comb[unlab_index_range]
                loss_ae_, _ = sess.run([loss_ae, train_op_ae], feed_dict={input_combined: X_ae})
                #print('Epoch Num {}, Batches Num {}, Loss_AE {}'.format
                      #k, i, np.round(loss_ae_, 3)))

            unlab_index_range = x_combined_index[(i + 1) * batch_size:]
            X_ae = Train_X_Comb[unlab_index_range]
            loss_ae_, _ = sess.run([loss_ae, train_op_ae], feed_dict={input_combined: X_ae})
            print('Epoch Num {}, Batches Num {}, Loss_AE {}'.format(k, i, np.round(loss_ae_, 3)))
        print('========================================================')
        print('End of training Autoencoder')
        print('========================================================')

        Val_X_latent = transfer_latent_space(Val_X, latent_combined, sess, input_combined)
        Test_X_latent = transfer_latent_space(Test_X, latent_combined, sess, input_combined)
        Train_X_latent = transfer_latent_space(Train_X, latent_combined, sess, input_combined)
        # Training classifier
        val_accuracy = {-2: 0, -1: 0}
        val_loss = {-2: 10, -1: 10}
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
            loss_val, acc_val = loss_acc_evaluation(Val_X_latent, Val_Y, input_latent, sess, loss_cls, accuracy_cls, true_label, k)
            val_loss.update({k: loss_val})
            val_accuracy.update({k: acc_val})
            print('====================================================')
            saver.save(sess, "/Conv-Semi-TF-PS/" + str(prop), global_step=k)
            if all([val_accuracy[k] < val_accuracy[k - 1], val_accuracy[k] < val_accuracy[k - 2]]):
                break
        print("Val Accuracy Over Epochs: ", val_accuracy)
        print("Val loss Over Epochs: ", val_loss)

        max_acc = max(val_accuracy.items(), key=lambda k: k[1])[0]
        save_path = "/Conv-Semi-TF-PS/" + str(prop) + '-' + str(max_acc)
        saver.restore(sess, save_path)

        y_pred = prediction_prob(Test_X_latent, classifier_output, input_latent, sess)
        test_acc = accuracy_score(Test_Y_ori, y_pred)
        f1_macro = f1_score(Test_Y_ori, y_pred, average='macro')
        f1_weight = f1_score(Test_Y_ori, y_pred, average='weighted')
        print('Conv-Semi 2-Steps test accuracy {}'.format(test_acc))

    return test_acc, f1_macro, f1_weight


def training_all_folds(label_proportions, num_filter):
    test_accuracy_fold = [[] for _ in range(len(label_proportions))]
    mean_std_acc = [[] for _ in range(len(label_proportions))]
    test_metrics_fold = [[] for _ in range(len(label_proportions))]
    mean_std_metrics = [[] for _ in range(len(label_proportions))]
    for index, prop in enumerate(label_proportions):
        for i in range(len(kfold_dataset)):
            test_accuracy, f1_macro, f1_weight = training(kfold_dataset[i], X_unlabeled=X_unlabeled, seed=7, prop=prop, num_filter=num_filter)
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

test_accuracy_fold, test_metrics_fold, mean_std_acc, mean_std_metrics = training_all_folds(label_proportions=[0.05],
                                                  num_filter=[32, 32, 64, 64])
a = 1





