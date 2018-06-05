import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import random
import pickle
import tensorflow as tf
import keras
# Settings
batch_size = 100
latent_dim = 800
units = 800  # num unit in the MLP hidden layer
num_dense = 0
kernel_size = (1, 3)
padding = 'same'
strides = 1
pool_size = (1, 2)
num_class = 5
initializer = tf.glorot_uniform_initializer()

filename = '../Mode-codes-Revised/paper2_data_for_DL_kfold_dataset.pickle'
with open(filename, 'rb') as f:
    kfold_dataset, _ = pickle.load(f)


def classifier(num_filter, input_labeled, num_dense):
    conv_layer = input_labeled
    for i in range(len(num_filter)):
        scope_name = 'encoder_set_' + str(i + 1)
        with tf.variable_scope(scope_name):
            conv_layer = tf.layers.conv2d(inputs=conv_layer, activation=tf.nn.relu, filters=num_filter[i],
                                                name='conv_1', kernel_size=kernel_size, strides=strides,
                                                padding=padding)
        if i % 2 != 0:
            conv_layer = tf.layers.max_pooling2d(conv_layer, pool_size=pool_size,
                                                          strides=pool_size, name='pool')

    dense = tf.layers.flatten(conv_layer)
    units = int(dense.get_shape().as_list()[-1] / 4)
    for i in range(num_dense):
        dense = tf.layers.dense(dense, units, activation=tf.nn.relu)
        units /= 2
    dense_last = dense
    dense = tf.layers.dropout(dense, 0.5)
    classifier_output = tf.layers.dense(dense, num_class, name='FC_4')
    return classifier_output


def cnn_model(input_labeled, true_label, num_filter):
    classifier_output = classifier(num_filter, input_labeled, num_dense)
    loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_label, logits=classifier_output),
                              name='loss_cls')
    train_op = tf.train.AdamOptimizer().minimize(loss_cls)

    correct_prediction = tf.equal(tf.argmax(true_label, 1), tf.argmax(classifier_output, 1))
    accuracy_cls = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return loss_cls, accuracy_cls, train_op, classifier_output


def loss_acc_evaluation(Test_X, Test_Y, sess, input_labeled, true_label, k, loss_cls, accuracy_cls):
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
    if len(Test_X_batch)>=1:
        loss_cls_, accuracy_cls_ = sess.run([loss_cls, accuracy_cls], feed_dict={input_labeled: Test_X_batch,
                                                   true_label: Test_Y_batch})
    metrics.append([loss_cls_, accuracy_cls_])
    mean_ = np.mean(np.array(metrics), axis=0)
    print('Epoch Num {}, Loss_cls_Val {}, Accuracy_Val {}'.format(k, mean_[0], mean_[1]))
    return mean_[0], mean_[1]


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
# ===================================


def training(one_fold, seed, prop, num_filter, epochs=20):
    Train_X = one_fold[0]
    Train_Y_ori = one_fold[1]
    Test_X = one_fold[2]
    Test_Y = one_fold[3]
    Test_Y_ori = one_fold[4]
    random.seed(seed)
    np.random.seed(seed)
    random_sample = np.random.choice(len(Train_X), size=round(prop*len(Train_X)), replace=False, p=None)
    Train_X = Train_X[random_sample]
    Train_Y_ori = Train_Y_ori[random_sample]
    #Train_X, Train_Y_ori = rand_stra_sample(Train_X, Train_Y_ori, prop)
    Train_X, Train_Y, Train_Y_ori, Val_X, Val_Y, Val_Y_ori = train_val_split(Train_X, Train_Y_ori)
    input_size = list(np.shape(Test_X)[1:])

    val_accuracy = {-2: 0, -1: 0}
    val_loss = {-2: 10, -1: 10}

    tf.reset_default_graph()
    with tf.Session() as sess:
        input_labeled = tf.placeholder(dtype=tf.float32, shape=[None] + input_size, name='input_labeled')
        true_label = tf.placeholder(tf.float32, shape=[None, num_class], name='true_label')
        loss_cls, accuracy_cls, train_op, classifier_output = cnn_model(input_labeled, true_label, num_filter)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=20)
        num_batches = len(Train_X) // batch_size
        for k in range(epochs):
            for i in range(num_batches):
                X_cls = Train_X[i * batch_size: (i + 1) * batch_size]
                Y_cls = Train_Y[i * batch_size: (i + 1) * batch_size]
                loss_cls_, accuracy_cls_, _ = sess.run([loss_cls, accuracy_cls, train_op],
                                                       feed_dict={input_labeled: X_cls, true_label: Y_cls})
                print('Epoch Num {}, Batches Num {}, Loss_cls {}, Accuracy_train {}'.format
                      (k, i, np.round(loss_cls_, 3), np.round(accuracy_cls_, 3)))

            X_cls = Train_X[(i + 1) * batch_size:]
            Y_cls = Train_Y[(i + 1) * batch_size:]
            loss_cls_, accuracy_cls_, _ = sess.run([loss_cls, accuracy_cls, train_op],
                                                   feed_dict={input_labeled: X_cls, true_label: Y_cls})
            print('Epoch Num {}, Batches Num {}, Loss_cls {}, Accuracy_train {}'.format
                  (k, i, np.round(loss_cls_, 3), np.round(accuracy_cls_, 3)))
            print('====================================================')
            loss_val, acc_val = loss_acc_evaluation(Val_X, Val_Y, sess, input_labeled, true_label, k, loss_cls, accuracy_cls)
            val_loss.update({k: loss_val})
            val_accuracy.update({k: acc_val})
            print('====================================================')
            saver.save(sess, "/Conv-Semi-TF-PS/" + str(prop), global_step=k)
            if all([val_accuracy[k] < val_accuracy[k - 1], val_accuracy[k] < val_accuracy[k - 2]]):
                break
        print("Val Accuracy Over Epochs: ", val_accuracy)
        print("Val Loss Over Epochs: ", val_loss)
        max_accuracy_val = max(val_accuracy.items(), key=lambda k: k[1])
        saver.restore(sess, "/Conv-Semi-TF-PS/" + str(prop) + '-' + str(max_accuracy_val[0]))

        y_pred = prediction_prob(Test_X, classifier_output, input_labeled, sess)
        test_acc = accuracy_score(Test_Y_ori, y_pred)
        f1_macro = f1_score(Test_Y_ori, y_pred, average='macro')
        f1_weight = f1_score(Test_Y_ori, y_pred, average='weighted')
        print('CNN Classifier test accuracy {}'.format(test_acc))
        return test_acc, f1_macro, f1_weight


def training_all_folds(label_proportions, num_filter):
    test_accuracy_fold = [[] for _ in range(len(label_proportions))]
    mean_std_acc = [[] for _ in range(len(label_proportions))]
    test_metrics_fold = [[] for _ in range(len(label_proportions))]
    mean_std_metrics = [[] for _ in range(len(label_proportions))]
    for index, prop in enumerate(label_proportions):
        for i in range(len(kfold_dataset)):
            test_accuracy, f1_macro, f1_weight = training(kfold_dataset[i], seed=7, prop=prop, num_filter=num_filter)
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

test_accuracy_fold, test_metrics_fold, mean_std_acc, mean_std_metrics = training_all_folds(label_proportions=[0.1, 0.25, 0.50, 0.75, 1.0],
                                                  num_filter=[32, 32, 64, 64, 128, 128])


