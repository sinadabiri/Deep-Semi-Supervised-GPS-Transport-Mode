import numpy as np
import random
import pickle
import tensorflow as tf
import keras
import time
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
start_time = time.clock()
# Settings
batch_size = 100
latent_dim = 800
change = 20
units = 400  # num unit in the MLP hidden layer
num_dense = 0
kernel_size = (1, 3)
padding = 'same'
strides = 1
pool_size = (1, 2)
num_class = 5
reg_l2 = tf.contrib.layers.l1_regularizer(scale=0.1)
initializer = tf.glorot_uniform_initializer()

# Import data
filename = '../Mode-codes-Revised/paper2_data_for_DL_kfold_dataset.pickle'
with open(filename, 'rb') as f:
    kfold_dataset, X_unlabeled = pickle.load(f)
a= 1
# Auto-Encoder Network


def autoencoder(input_combined, input_labeled, num_filter, num_dense):
    encoded_combined = input_combined
    encoded_labeled = input_labeled
    ae_in_out = []
    for i in range(len(num_filter)):
        name = 'input_dec'+str(i+1)
        input_ = tf.placeholder(dtype=tf.float32, shape=encoded_combined.get_shape().as_list(), name=name)
        scope_name = 'encoder_set_' + str(i + 1)
        with tf.variable_scope(scope_name, regularizer=reg_l2):
            encoded_combined = tf.layers.conv2d(inputs=input_, activation=tf.nn.relu, filters=num_filter[i],
                                                    name='conv_1', kernel_size=kernel_size, strides=strides,padding=padding)
        with tf.variable_scope(scope_name, reuse=True, regularizer=reg_l2):
            encoded_labeled = tf.layers.conv2d(inputs=encoded_labeled, activation=tf.nn.relu, filters=num_filter[i],
                                                name='conv_1', kernel_size=kernel_size, strides=strides, padding=padding)
        activation = tf.nn.sigmoid if i == 0 else tf.nn.relu
        decoder = tf.layers.conv2d_transpose(inputs=encoded_combined, activation=activation,
                                                          filters=input_.get_shape().as_list()[-1],
                                                          kernel_size=kernel_size,
                                                          strides=strides, padding=padding, kernel_regularizer=reg_l2)

        if i % 2 != 0:
            encoded_combined = tf.layers.max_pooling2d(encoded_combined, pool_size=pool_size,
                                                          strides=pool_size, name='pool')
            encoded_labeled = tf.layers.max_pooling2d(encoded_labeled, pool_size=pool_size,
                                                          strides=pool_size, name='pool')
        ae_in_out.append([input_, encoded_combined, decoder])

    encoded_combined = tf.layers.flatten(encoded_combined)
    ae_in_out[-1][1] = encoded_combined
    encoded_labeled = tf.layers.flatten(encoded_labeled)
    #input_cls = tf.placeholder(dtype=tf.float32, shape=encoded_combined.get_shape().as_list())
    units = int(encoded_combined.get_shape().as_list()[-1]/4)
    for i in range(num_dense):
        input_ = tf.placeholder(dtype=tf.float32, shape=encoded_combined.get_shape().as_list(), name=name)
        scope_name = 'dense_set' + str(i+1)
        with tf.variable_scope(scope_name, regularizer=reg_l2):
            encoded_combined = tf.layers.dense(input_, units, activation=tf.nn.relu, name='FC_1')
        with tf.variable_scope(scope_name, reuse=True, regularizer=reg_l2):
            encoded_labeled = tf.layers.dense(encoded_labeled, units, activation=tf.nn.relu, name='FC_1')

        decoder = tf.layers.dense(encoded_combined, input_.get_shape().as_list()[-1], activation=tf.nn.relu, kernel_regularizer=reg_l2)
        ae_in_out.append([input_, encoded_combined, decoder])
        units /= 2
    dense = encoded_labeled

    scope_name = 'last_FC'
    input_ = tf.placeholder(dtype=tf.float32, shape=encoded_combined.get_shape().as_list(), name=name)
    with tf.variable_scope(scope_name, regularizer=reg_l2):
        encoded_combined = tf.layers.dense(input_, num_class, activation=tf.nn.relu, name='FC_1')
    with tf.variable_scope(scope_name, reuse=True, regularizer=reg_l2):
        encoded_labeled = tf.layers.dense(encoded_labeled, num_class, activation=None, name='FC_1')

    #decoder = tf.layers.dense(encoded_combined, input_.get_shape().as_list()[-1], activation=tf.nn.relu, kernel_regularizer=reg_l2)
    #ae_in_out.append([input_, encoded_combined, decoder])
    encoded_labeled = tf.layers.dropout(encoded_labeled, 0.5)
    classifier_output = encoded_labeled

    return ae_in_out, classifier_output, dense


def semi_supervised(input_labeled, input_combined, true_label, alpha, beta, num_class, latent_dim, num_filter):
    ae_in_out, classifier_output, dense = autoencoder(input_combined=input_combined, input_labeled=input_labeled, num_filter=num_filter, num_dense=num_dense)
    #classifier_output = classifier_cnn(latent_labeled, num_filter=num_filter)
    loss_ae = []
    train_op_ae = []
    inputs_ae = []
    encoders_ae = []
    loss_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'EasyNet'))
    for ae in ae_in_out:
        inputs_ae.append(ae[0])
        encoders_ae.append(ae[1])
        loss_ae_ = (tf.reduce_mean(tf.square(ae[0] - ae[2]), name='loss_ae') + loss_reg)* 100
        loss_ae.append(loss_ae_)
        train_op_ae.append(tf.train.AdamOptimizer().minimize(loss_ae_))

    loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_label, logits=classifier_output),
                              name='loss_cls') + loss_reg

    train_op_cls = tf.train.AdamOptimizer().minimize(loss_cls)

    # train_op = train_op = tf.layers.optimize_loss(total_loss, optimizer='Adam')

    correct_prediction = tf.equal(tf.argmax(true_label, 1), tf.argmax(classifier_output, 1))
    accuracy_cls = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return loss_ae, loss_cls, accuracy_cls, train_op_ae, train_op_cls, classifier_output, dense, inputs_ae, encoders_ae


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


def loss_acc_evaluation(Test_X, Test_Y, sess, loss_cls, accuracy_cls, input_labeled, true_label, k, batch_size):
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
    print('Epoch Num {}, Loss_cls_val {}, Accuracy_val {}'.format(k, mean_[0], mean_[1]))
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


def training(one_fold, X_unlabeled, seed, prop, num_filter, epochs_ae=10, epochs_cls=20):
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

        loss_ae, loss_cls, accuracy_cls, train_op_ae, train_op_cls, classifier_output, dense, inputs_ae, encoders_ae = semi_supervised(
            input_labeled=input_labeled, input_combined=input_combined,
            true_label=true_label, alpha=alpha, beta=beta,
            num_class=num_class,
            latent_dim=latent_dim, num_filter=num_filter)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=20)

        num_batches = len(Train_X_Comb) // batch_size
        for index, train_op in enumerate(train_op_ae):
            input_ = inputs_ae[index]
            for k in range(epochs_ae):
                x_combined_index = get_combined_index(train_x_comb=Train_X_Comb)
                for i in range(num_batches):
                    unlab_index_range = x_combined_index[i * batch_size: (i + 1) * batch_size]
                    X_ae = Train_X_Comb[unlab_index_range]
                    loss_ae_, _ = sess.run([loss_ae[index], train_op], feed_dict={input_: X_ae})
                    #print('AE Num {}, Epoch Num {}, Batches Num {}, Loss_AE {}'.format
                          #(index, k, i, np.round(loss_ae_, 3)))

                unlab_index_range = x_combined_index[(i + 1) * batch_size:]
                X_ae = Train_X_Comb[unlab_index_range]
                loss_ae_, _ = sess.run([loss_ae[index], train_op], feed_dict={input_: X_ae})
                print('AE Num {}, Epoch Num {}, Batches Num {}, Loss_AE {}'.format
                      (index, k, i, np.round(loss_ae_, 3)))

            # Transfer X_Comb_New to the next layer
            X_Comb_New = np.zeros([len(Train_X_Comb)] + encoders_ae[index].get_shape().as_list()[1:])
            for i in range(num_batches):
                X_Comb_New[i * batch_size: (i + 1) * batch_size] = sess.run(encoders_ae[index], feed_dict={
                    input_: Train_X_Comb[i * batch_size: (i + 1) * batch_size]})
            X_Comb_New[(i + 1) * batch_size:] = sess.run(encoders_ae[index],
                                                         feed_dict={input_: Train_X_Comb[(i + 1) * batch_size:]})
            Train_X_Comb = X_Comb_New

        print('========================================================')
        print('End of pre-training using Autoencoders')
        print('========================================================')

        # Training classifier
        val_accuracy = {-2: 0, -1: 0}
        val_loss = {-2: 10, -1: 10}
        for k in range(epochs_cls):
            num_batches = len(Train_X) // batch_size
            x_labeled_index = np.arange(len(Train_X))
            for i in range(num_batches):
                lab_index_range = x_labeled_index[i * batch_size: (i + 1) * batch_size]
                X_cls = Train_X[lab_index_range]
                Y_cls = Train_Y[lab_index_range]
                loss_cls_, accuracy_cls_, _ = sess.run([loss_cls, accuracy_cls, train_op_cls],
                                                       feed_dict={input_labeled: X_cls, true_label: Y_cls})
                #print('Epoch Num {}, Batches Num {}, Loss_cls {}, Accuracy_train {}'.format
                      #(k, i, np.round(loss_cls_, 3), np.round(accuracy_cls_, 3)))

            lab_index_range = x_labeled_index[(i + 1) * batch_size:]
            X_cls = Train_X[lab_index_range]
            Y_cls = Train_Y[lab_index_range]
            loss_cls_, accuracy_cls_, _ = sess.run([loss_cls, accuracy_cls, train_op_cls],
                                                   feed_dict={input_labeled: X_cls, true_label: Y_cls})
            print('Epoch Num {}, Batches Num {}, Loss_cls {}, Accuracy_train {}'.format
                  (k, i, np.round(loss_cls_, 3), np.round(accuracy_cls_, 3)))
            print('====================================================')
            loss_val, acc_val = loss_acc_evaluation(Val_X, Val_Y, sess, loss_cls, accuracy_cls, input_labeled, true_label, k, batch_size)
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

        y_pred = prediction_prob(Test_X, classifier_output, input_labeled, sess)
        test_acc = accuracy_score(Test_Y_ori, y_pred)
        f1_macro = f1_score(Test_Y_ori, y_pred, average='macro')
        f1_weight = f1_score(Test_Y_ori, y_pred, average='weighted')
        print('Conv-Semi Layer-Wise test accuracy {}'.format(test_acc))

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
        print('Semi-Layer-Wise test accuracy for prop {}: Mean {}, std {}'.format(prop, mean_std_acc[index][0], mean_std_acc[index][1]))
        print('Semi-Layer-Wise test metrics for prop {}: Mean {}, std {}'.format(prop, mean_std_metrics[index][0], mean_std_metrics[index][1]))
        print('\n')
    return test_accuracy_fold, test_metrics_fold, mean_std_acc, mean_std_metrics

test_accuracy_fold, test_metrics_fold, mean_std_acc, mean_std_metrics = training_all_folds(label_proportions=[0.05, 0.15, 0.35],
                                                  num_filter=[32, 32, 64, 64])
a = 1


