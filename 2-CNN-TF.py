import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import random
import pickle
import tensorflow as tf

# Settings
prop = 0.05
new_channel = 3
batch_size = 32
latent_dim = 800
epochs = 20
units = 800  # num unit in the MLP hidden layer
num_filter = [32, 64, 128]
kernel_size = (1, 3)
padding = 'same'
strides = 1
pool_size = (1, 2)
num_class = 5
initializer = tf.glorot_uniform_initializer()

filename = '../Mode-codes-Revised/paper2_data_for_DL_train_val_test_prepared.pickle'
with open(filename, 'rb') as f:
    Train_X, Train_Y, Val_X, Val_Y, Val_Y_ori, Test_X, Test_Y, Test_Y_ori, X_unlabeled = pickle.load(f)
# Training and test set for GPS segments
random.seed(7)
np.random.seed(7)
tf.set_random_seed(7)
index = np.arange(len(Train_X))
np.random.shuffle(index)
Train_X = Train_X[index[:round(prop*len(Train_X))]]
Train_Y = Train_Y[index[:round(prop*len(Train_Y))]]
input_size = list(np.shape(Test_X)[1:])


def classifier(num_filter, input_labeled):
    for i in range(len(num_filter)):
        if i != 0:
            scope_name = 'encoder_set_' + str(i + 1)
            with tf.variable_scope(scope_name, initializer=tf.glorot_uniform_initializer()):
                conv_layer = tf.layers.conv2d(inputs=conv_layer, activation=tf.nn.relu, filters=num_filter[i],
                                                    name='conv_1', kernel_size=kernel_size, strides=strides,padding=padding)
                conv_layer = tf.layers.conv2d(inputs=conv_layer, activation=tf.nn.relu, filters=num_filter[i],
                                                    name='conv_2', kernel_size=kernel_size, strides=strides, padding=padding)
                conv_layer = tf.layers.max_pooling2d(conv_layer, pool_size=pool_size,
                                                           strides=pool_size,name='pool')
        else:
            scope_name = 'encoder_set_' + str(i + 1)
            with tf.variable_scope(scope_name, initializer=tf.glorot_uniform_initializer()):
                conv_layer = tf.layers.conv2d(inputs=input_labeled, activation=tf.nn.relu, filters=num_filter[i],
                                                    name='conv_1', kernel_size=kernel_size, strides=strides, padding=padding)
                conv_layer = tf.layers.conv2d(inputs=conv_layer, activation=tf.nn.relu, filters=num_filter[i],
                                                    name='conv_2', kernel_size=kernel_size, strides=strides, padding=padding)
                conv_layer = tf.layers.max_pooling2d(conv_layer, pool_size=pool_size, padding=padding, strides=pool_size,
                                                           name='pool')

    conv_layer = tf.layers.dropout(conv_layer, 0.5)
    flat = tf.layers.flatten(conv_layer)
    scope_name = 'dense_set_'
    with tf.variable_scope(scope_name, initializer=tf.glorot_uniform_initializer()):
        dense = tf.layers.dense(flat, int(flat.get_shape().as_list()[1]/4), name='dense_1', activation=tf.nn.relu)
        dense = tf.layers.dropout(dense, 0.5)
        classifier_output = tf.layers.dense(dense, num_class, name='FC_2')
    return classifier_output


def cnn_model(input_labeled, true_label, num_filter):
    classifier_output = classifier(num_filter, input_labeled)
    loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_label, logits=classifier_output),
                              name='loss_cls')
    train_op = tf.train.AdamOptimizer().minimize(loss_cls)

    correct_prediction = tf.equal(tf.argmax(true_label, 1), tf.argmax(classifier_output, 1))
    accuracy_cls = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return loss_cls, accuracy_cls, train_op, classifier_output


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
    if len(Test_X_batch)>=1:
        loss_cls_, accuracy_cls_ = sess.run([loss_cls, accuracy_cls], feed_dict={input_labeled: Test_X_batch,
                                                   true_label: Test_Y_batch})
    metrics.append([loss_cls_, accuracy_cls_])
    mean_ = np.mean(np.array(metrics), axis=0)
    print('Epoch Num {}, Loss_cls_Val {}, Accuracy_Val {}'.format(k, mean_[0], mean_[1]))
    return mean_[0], mean_[1]
# ===================================

# Create CNN classifier
input_labeled = tf.placeholder(dtype=tf.float32, shape=[None]+input_size, name='input_labeled')
true_label = tf.placeholder(tf.float32, shape=[None, num_class], name='true_label')

conv_layer = tf.layers.conv2d(inputs=input_labeled, activation=tf.nn.relu, filters=32,kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=initializer)
conv_layer = tf.layers.conv2d(inputs=conv_layer, activation=tf.nn.relu, filters=32, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=initializer)
conv_layer = tf.layers.max_pooling2d(conv_layer, pool_size=pool_size, padding=padding, strides=pool_size, name='pool')

conv_layer = tf.layers.conv2d(inputs=conv_layer, activation=tf.nn.relu, filters=64, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=initializer)
conv_layer = tf.layers.conv2d(inputs=conv_layer, activation=tf.nn.relu, filters=64, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=initializer)
conv_layer = tf.layers.max_pooling2d(conv_layer, pool_size=pool_size, padding=padding, strides=pool_size, name='pool')

conv_layer = tf.layers.conv2d(inputs=conv_layer, activation=tf.nn.relu, filters=128, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=initializer)
conv_layer = tf.layers.conv2d(inputs=conv_layer, activation=tf.nn.relu, filters=128, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=initializer)
conv_layer = tf.layers.max_pooling2d(conv_layer, pool_size=pool_size, padding=padding, strides=pool_size, name='pool')

conv_layer = tf.layers.flatten(conv_layer)

#conv_layer = tf.layers.dense(conv_layer, int(conv_layer.get_shape().as_list()[1]/4), activation=tf.nn.relu, kernel_initializer=initializer)
#conv_layer = tf.layers.dense(conv_layer, int(conv_layer.get_shape().as_list()[1]/2), activation=tf.nn.relu, kernel_initializer=initializer)
#conv_layer = tf.layers.dense(conv_layer, int(conv_layer.get_shape().as_list()[1]/2), activation=tf.nn.relu, kernel_initializer=initializer)

conv_layer = tf.layers.dropout(conv_layer, 0.5)
classifier_output = tf.layers.dense(conv_layer, num_class, kernel_initializer=initializer)

loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_label, logits=classifier_output), name='loss_cls')
train_op = tf.train.AdamOptimizer().minimize(loss_cls)

correct_prediction = tf.equal(tf.argmax(true_label, 1), tf.argmax(classifier_output, 1))
accuracy_cls = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

val_loss = {}
val_accuracy = {}
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=15)
    num_batches = len(Train_X) // batch_size
    for k in range(epochs):
        for i in range(num_batches):
            X_cls = Train_X[i * batch_size: (i + 1) * batch_size]
            Y_cls = Train_Y[i * batch_size: (i + 1) * batch_size]
            loss_cls_, accuracy_cls_, _ = sess.run([loss_cls, accuracy_cls, train_op], feed_dict={input_labeled: X_cls, true_label: Y_cls})
            print('Epoch Num {}, Batches Num {}, Loss_cls {}, Accuracy_train {}'.format
                  (k, i, np.round(loss_cls_, 3), np.round(accuracy_cls_, 3)))

        X_cls = Train_X[(i + 1) * batch_size:]
        Y_cls = Train_Y[(i + 1) * batch_size:]
        loss_cls_, accuracy_cls_, _ = sess.run([loss_cls, accuracy_cls, train_op],
                                               feed_dict={input_labeled: X_cls, true_label: Y_cls})
        print('Epoch Num {}, Batches Num {}, Loss_cls {}, Accuracy_train {}'.format
              (k, i, np.round(loss_cls_, 3), np.round(accuracy_cls_, 3)))
        print('====================================================')
        loss_val, acc_val = loss_acc_evaluation(Val_X, Val_Y)
        val_loss.update({k: loss_val})
        val_accuracy.update({k: acc_val})
        print('====================================================')
        saver.save(sess, "/Conv-Semi-TF-PS/" + str(prop), global_step=k)

    print("Val Accuracy Over Epochs: ", val_accuracy)
    print("Val Loss Over Epochs: ", val_loss)
    max_accuracy_val = max(val_accuracy.items(), key=lambda k: k[1])
    print('Max accuracy ', max_accuracy_val)
    saver.restore(sess, "/Conv-Semi-TF-PS/" + str(prop) + '-' + str(max_accuracy_val[0]))
    print('\n')
    print('====================================================')
    loss_Test, acc_Test = loss_acc_evaluation(Test_X, Test_Y)
    print('CNN classifier test-loss {} and CNN classifier test-accuracy {}'.format(loss_Test, acc_Test))
    print('====================================================')

    # Another way to compute test accuracy.
    y_pred = sess.run(tf.argmax(input=classifier_output, axis=1), feed_dict={input_labeled: Test_X})
    print('Test Accuracy %: ', accuracy_score(Test_Y_ori, y_pred))
    print('\n')
    print('Confusin matrix: ', confusion_matrix(Test_Y_ori, y_pred))
    print('\n')
    print(classification_report(Test_Y_ori, y_pred, digits=3))


