import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import random
import pickle
import tensorflow as tf
import keras

# Settings
latent_dim = 800
batch_size = 100
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

filename = '../Mode-codes-Revised/paper2_data_for_DL_kfold_dataset.pickle'
with open(filename, 'rb') as f:
    kfold_dataset, X_unlabeled = pickle.load(f)

# Encoder Network


def encoder_network(latent_dim, num_filter, input_combined, input_unlabeled, input_labeled):
    encoded_combined = input_combined
    encoded_labeled = input_labeled
    encoded_unlabeled = input_unlabeled
    layers_shape = []
    for i in range(len(num_filter)):
        scope_name = 'encoder_set_' + str(i + 1)
        with tf.variable_scope(scope_name):
            encoded_combined = tf.layers.conv2d(inputs=encoded_combined, activation=tf.nn.relu, filters=num_filter[i],
                                                name='conv_1', kernel_size=kernel_size, strides=strides, padding=padding)
        with tf.variable_scope(scope_name, reuse=True):
            encoded_labeled = tf.layers.conv2d(inputs=encoded_labeled, activation=tf.nn.relu, filters=num_filter[i],
                                               name='conv_1', kernel_size=kernel_size, strides=strides, padding=padding)
        with tf.variable_scope(scope_name, reuse=True):
            encoded_unlabeled = tf.layers.conv2d(inputs=encoded_unlabeled, activation=tf.nn.relu, filters=num_filter[i],
                                               name='conv_1', kernel_size=kernel_size, strides=strides, padding=padding)

        if i % 2 != 0:
            encoded_combined = tf.layers.max_pooling2d(encoded_combined, pool_size=pool_size,
                                                          strides=pool_size, name='pool')
            encoded_labeled = tf.layers.max_pooling2d(encoded_labeled, pool_size=pool_size,
                                                          strides=pool_size, name='pool')
            encoded_unlabeled = tf.layers.max_pooling2d(encoded_unlabeled, pool_size=pool_size,
                                                      strides=pool_size, name='pool')
        layers_shape.append(encoded_combined.get_shape().as_list())

    layers_shape.append(encoded_combined.get_shape().as_list())
    latent_combined = encoded_combined
    latent_labeled = encoded_labeled
    latent_unlabled = encoded_unlabeled

    return latent_combined, latent_labeled, latent_unlabled, layers_shape

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


def classifier_mlp(latent_labeled, latent_unlabeled, num_class, num_fliter_cls, num_dense, num_filter):
    conv_layer_l = latent_labeled
    conv_layer_ul = latent_unlabeled

    for i in range(len(num_fliter_cls)):
        scope_name = 'cls_set_' + str(i + 1)
        with tf.variable_scope(scope_name):
            conv_layer_l = tf.layers.conv2d(inputs=conv_layer_l, activation=tf.nn.relu, filters=num_filter_cls[i],
                                  kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=initializer, name='conv')
        with tf.variable_scope(scope_name, reuse=True):
            conv_layer_ul = tf.layers.conv2d(inputs=conv_layer_ul, activation=tf.nn.relu, filters=num_filter_cls[i],
                                          kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=initializer, name='conv')
        if len(num_filter) % 2 == 0:
            if i % 2 != 0:
                conv_layer_l = tf.layers.max_pooling2d(conv_layer_l, pool_size=pool_size,strides=pool_size, name='pool')
                conv_layer_ul = tf.layers.max_pooling2d(conv_layer_ul, pool_size=pool_size,strides=pool_size, name='pool')

        else:
            if i % 2 == 0:
                conv_layer_l = tf.layers.max_pooling2d(conv_layer_l, pool_size=pool_size,strides=pool_size, name='pool')
                conv_layer_ul = tf.layers.max_pooling2d(conv_layer_ul, pool_size=pool_size,strides=pool_size, name='pool')

    dense_l = tf.layers.flatten(conv_layer_l)
    dense_ul = tf.layers.flatten(conv_layer_ul)
    units = int(dense_l.get_shape().as_list()[-1] / 4)
    for i in range(num_dense):
        scope_name = 'dense_set_' + str(i + 1)
        with tf.variable_scope(scope_name):
            dense_l = tf.layers.dense(dense_l, units, activation=tf.nn.relu)
        with tf.variable_scope(scope_name, reuse=True):
            dense_ul = tf.layers.dense(dense_ul, units, activation=tf.nn.relu)

        units /= 2
    dense_last = dense_l
    dense_l = tf.layers.dropout(dense_l, 0.5)
    dense_ul = tf.layers.dropout(dense_ul, 0.5)
    scope_name = 'FC_set_'
    with tf.variable_scope(scope_name):
        classifier_output_l = tf.layers.dense(dense_l, num_class, name='FC_4')
    with tf.variable_scope(scope_name, reuse=True):
        classifier_output_ul = tf.layers.dense(dense_ul, num_class, name='FC_4')
    return classifier_output_l, classifier_output_ul, dense_last


def semi_supervised(input_labeled, input_unlabeled, input_combined, true_label_l, true_label_ul, alpha, gama, beta, alpha_cls, beta_cls, num_class, latent_dim, num_filter, input_size):
    latent_combined, latent_labeled, latent_unlabled, layers_shape = encoder_network(latent_dim, num_filter, input_combined=input_combined, input_unlabeled=input_unlabeled, input_labeled=input_labeled)
    decoded_output = decoder_network(latent_combined=latent_combined, input_size=input_size, kernel_size=kernel_size, padding=padding, activation=activation, num_filter=num_filter)
    classifier_output_l, classifier_output_ul, dense_last = classifier_mlp(latent_labeled=latent_labeled, latent_unlabeled=latent_unlabled, num_class=num_class, num_fliter_cls=num_filter_cls, num_dense=num_dense, num_filter=num_filter)

    loss_ae = tf.reduce_mean(tf.square(input_combined - decoded_output), name='loss_ae') * 100
    loss_cls_l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_label_l, logits=classifier_output_l),
                              name='loss_cls')
    loss_cls_ul = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_label_ul, logits=classifier_output_ul),
                              name='loss_cls_ul')
    total_loss_ae_cls = alpha*loss_ae + beta*loss_cls_l
    #total_loss_cls = alpha_cls * loss_cls_ul + beta_cls * loss_cls_l
    total_loss_cls = alpha * loss_cls_ul + beta * loss_cls_l
    total_loss_all = alpha*loss_ae + gama*loss_cls_ul + beta*loss_cls_l
    loss_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'EasyNet'))
    train_op_ae = tf.train.AdamOptimizer().minimize(loss_ae)
    train_op_ae_cls = tf.train.AdamOptimizer().minimize(total_loss_ae_cls)
    train_op_cls = tf.train.AdamOptimizer().minimize(total_loss_cls)
    train_op_all = tf.train.AdamOptimizer().minimize(total_loss_all)
    # train_op = train_op = tf.layers.optimize_loss(total_loss, optimizer='Adam')

    correct_prediction_l = tf.equal(tf.argmax(true_label_l, 1), tf.argmax(classifier_output_l, 1))
    accuracy_cls_l = tf.reduce_mean(tf.cast(correct_prediction_l, tf.float32))
    correct_prediction_ul = tf.equal(tf.argmax(true_label_ul, 1), tf.argmax(classifier_output_ul, 1))
    accuracy_cls_ul = tf.reduce_mean(tf.cast(correct_prediction_ul, tf.float32))
    return loss_ae, loss_cls_l, loss_cls_ul, total_loss_ae_cls, total_loss_cls, total_loss_all, train_op_ae, train_op_ae_cls, train_op_cls, train_op_all, accuracy_cls_l, accuracy_cls_ul, classifier_output_ul, classifier_output_l


def get_combined_index(train_x_comb):
    np.random.seed(np.random.randint(1, 10, size=1)[0])
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


def pseudo_label(X_ul, sess, classifier_output_ul, input_unlabeled):
    prediction = sess.run(tf.nn.softmax(classifier_output_ul), feed_dict={input_unlabeled: X_ul})
    #Y_ul = sess.run(tf.one_hot(prediction, depth=num_class))
    return prediction

def loss_acc_evaluation(Test_X, Test_Y, loss_cls_l, accuracy_cls_l, input_labeled, true_label_l, k, sess):
    metrics = []
    for i in range(len(Test_X) // batch_size):
        Test_X_batch = Test_X[i * batch_size:(i + 1) * batch_size]
        Test_Y_batch = Test_Y[i * batch_size:(i + 1) * batch_size]
        loss_cls_, accuracy_cls_ = sess.run([loss_cls_l, accuracy_cls_l],
                                            feed_dict={input_labeled: Test_X_batch,
                                                       true_label_l: Test_Y_batch})
        metrics.append([loss_cls_, accuracy_cls_])
    Test_X_batch = Test_X[(i + 1) * batch_size:]
    Test_Y_batch = Test_Y[(i + 1) * batch_size:]
    if len(Test_X_batch) >= 1:
        loss_cls_, accuracy_cls_ = sess.run([loss_cls_l, accuracy_cls_l],
                                        feed_dict={input_labeled: Test_X_batch,
                                                   true_label_l: Test_Y_batch})
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
    Train_X_Comb = X_unlabeled
    Train_X_Unlabel = X_unlabeled
    input_size = list(np.shape(Test_X)[1:])

    tf.reset_default_graph()
    with tf.Session() as sess:
        input_labeled = tf.placeholder(dtype=tf.float32, shape=[None] + input_size, name='input_labeled')
        input_unlabeled = tf.placeholder(dtype=tf.float32, shape=[None] + input_size, name='input_unlabeled')
        input_combined = tf.placeholder(dtype=tf.float32, shape=[None] + input_size, name='input_combined')
        true_label_l = tf.placeholder(tf.float32, shape=[None, num_class], name='true_label_l')
        true_label_ul = tf.placeholder(tf.float32, shape=[None, num_class], name='true_label_ul')

        alpha = tf.placeholder(tf.float32, shape=(), name='alpha')
        gama = tf.placeholder(tf.float32, shape=(), name='gama')
        beta = tf.placeholder(tf.float32, shape=(), name='beta')
        alpha_cls = tf.placeholder(tf.float32, shape=(), name='alpha_cls')
        beta_cls = tf.placeholder(tf.float32, shape=(), name='beta_cls')

        loss_ae, loss_cls_l, loss_cls_ul, total_loss_ae_cls, total_loss_cls, total_loss_all, train_op_ae, train_op_ae_cls, train_op_cls, train_op_all, accuracy_cls_l, accuracy_cls_ul, classifier_output_ul, classifier_output_l \
            = semi_supervised(input_labeled, input_unlabeled, input_combined, true_label_l, true_label_ul, alpha, gama, beta, alpha_cls, beta_cls, num_class, latent_dim, num_filter, input_size)

        num_batches = len(Train_X_Comb) // batch_size

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=20)
        '''
        for k in range(epochs_ae):
            num_batches = len(Train_X_Comb) // batch_size
            x_combined_index = get_combined_index(train_x_comb=Train_X_Comb)
            for i in range(num_batches):
                unlab_index_range = x_combined_index[i * batch_size: (i + 1) * batch_size]
                X_ae = Train_X_Comb[unlab_index_range]
                loss_ae_, _ = sess.run([loss_ae, train_op_ae], feed_dict={input_combined: X_ae})
                #print('Epoch Num {}, Batches Num {}, Loss_AE {}'.format
                      #(k, i, np.round(loss_ae_, 3)))

            unlab_index_range = x_combined_index[(i + 1) * batch_size:]
            X_ae = Train_X_Comb[unlab_index_range]
            loss_ae_, _ = sess.run([loss_ae, train_op_ae], feed_dict={input_combined: X_ae})
            print('Epoch Num {}, Batches Num {}, Loss_AE {}'.format(k, i, np.round(loss_ae_, 3)))
        print('========================================================')
        print('End of training Autoencoder')
        print('========================================================')
        '''
        # Train for pseudo labeling
        val_accuracy = {-2: 0, -1: 0}
        val_loss = {-2: 10, -1: 10}
        alfa_val = 1
        beta_val = 1
        change_to_ae = 1  # the value defines that algorithm is ready to change to joint ae-cls
        change_times = 0  # No. of times change from cls to ae-cls
        for k in range(epochs_cls):
            x_unlabeled_index = get_combined_index(train_x_comb=Train_X_Unlabel)
            x_labeled_index = get_labeled_index(train_x_comb=Train_X_Unlabel, train_x=Train_X)
            # x_labeled_index = np.arange(len(Train_X))
            for i in range(num_batches):
                unlab_index_range = x_unlabeled_index[i * batch_size: (i + 1) * batch_size]
                lab_index_range = x_labeled_index[i * batch_size: (i + 1) * batch_size]
                X_ul = Train_X_Unlabel[unlab_index_range]
                Y_ul = pseudo_label(X_ul, sess, classifier_output_ul, input_unlabeled)
                X_l = Train_X[lab_index_range]
                Y_l = Train_Y[lab_index_range]
                accuracy_cls_l_, _ = sess.run([accuracy_cls_l, train_op_cls],
                                              feed_dict={alpha: alfa_val, beta: beta_val, input_labeled: X_l,
                                                         input_unlabeled: X_ul, true_label_l: Y_l, true_label_ul: Y_ul})
                #print('Epoch Num {}, Batches Num {}, accuracy_cls_l {}'.format
                      #(k, i, accuracy_cls_l_))

            unlab_index_range = x_unlabeled_index[(i + 1) * batch_size:]
            lab_index_range = x_labeled_index[(i + 1) * batch_size:]
            X_ul = Train_X_Unlabel[unlab_index_range]
            Y_ul = pseudo_label(X_ul, sess, classifier_output_ul, input_unlabeled)
            X_l = Train_X[lab_index_range]
            Y_l = Train_Y[lab_index_range]
            accuracy_cls_l_, _ = sess.run([accuracy_cls_l, train_op_cls], feed_dict={alpha: alfa_val, beta: beta_val,
                                                                                     input_labeled: X_l,
                                                                                     input_unlabeled: X_ul,
                                                                                     true_label_l: Y_l,
                                                                                     true_label_ul: Y_ul})
            print('Epoch Num {}, Batches Num {}, accuracy_cls_l {}'.format
                  (k, i, accuracy_cls_l_))
            print('====================================================')
            loss_val, acc_val = loss_acc_evaluation(Val_X, Val_Y, loss_cls_l, accuracy_cls_l, input_labeled, true_label_l, k, sess)
            val_loss.update({k: loss_val})
            val_accuracy.update({k: acc_val})
            print('====================================================')
            saver.save(sess, "/Conv-Semi-TF-PS/" + str(prop), global_step=k)
            # save_path = "/Conv-Semi/" + str(prop) + '/' + str(k) + ".ckpt"
            # checkpoint = os.path.join(os.getcwd(), save_path)
            # saver.save(sess, checkpoint)
            # if alfa_val == 1:
            # beta_val += 0.05

            if all([change_to_ae, val_accuracy[k] < val_accuracy[k - 1], val_accuracy[k] < val_accuracy[k - 2]]):
                # save_path = "/Conv-Semi/" + str(prop) + '/' + str(k-1) + ".ckpt"
                # checkpoint = os.path.join(os.getcwd(), save_path)
                max_acc = max(val_accuracy.items(), key=lambda k: k[1])[0]
                save_path = "/Conv-Semi-TF-PS/" + str(prop) + '-' + str(max_acc)
                saver.restore(sess, save_path)
                alfa_val = 1.5
                beta_val = 0.1
                change_times += 1
                change_to_ae = 1
                key = 'change_' + str(k)
                val_accuracy.update({key: val_accuracy[k]})
                val_loss.update({key: val_loss[k]})


            elif all([not change_to_ae, val_accuracy[k] < val_accuracy[k - 1],
                      val_accuracy[k] < val_accuracy[k - 2]]):
                # save_path = "/Conv-Semi/" + str(prop) + '/' + str(k - 1) + ".ckpt"
                # #checkpoint = os.path.join(os.getcwd(), save_path)
                max_acc = max(val_accuracy.items(), key=lambda k: k[1])[0]
                # saver.restore(sess, "/Conv-Semi-TF-PS/" + str(prop) + '/' + str(max_acc) + ".ckpt")
                save_path = "/Conv-Semi-TF-PS/" + str(prop) + '-' + str(max_acc)
                saver.restore(sess, save_path)
                alfa_val = 1
                beta_val = 0.2
                change_to_ae = 1
                key = 'change_' + str(k)
                val_accuracy.update({key: val_accuracy[k]})
                val_loss.update({key: val_loss[k]})

            if change_times == 2:
                break

        print("Val_Accu ae+cls Over Epochs: ", val_accuracy)
        print("Val_loss ae+cls Over Epochs: ", val_loss)
        print('====================================================')
        y_pred = prediction_prob(Test_X, classifier_output_l, input_labeled, sess)
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

test_accuracy_fold, test_metrics_fold, mean_std_acc, mean_std_metrics = training_all_folds(label_proportions=[0.15, 0.35],
                                                  num_filter=[32, 32, 64, 64])
a = 1



