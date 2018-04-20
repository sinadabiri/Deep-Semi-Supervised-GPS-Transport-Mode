import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
import pickle
from keras.optimizers import Adam
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import time
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
x = np.ones((1, 2, 3))
a = np.transpose(x, (1, 0, 2))

tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
sess = print(tf.Session(config=tf.ConfigProto(log_device_placement=True)))

start_time = time.clock()
np.random.seed(7)
random.seed(7)

filename = '../Mode-codes-Revised/paper2_data_for_DL_train_val_test_prepared.pickle'
with open(filename, 'rb') as f:
    Train_X, Train_Y, Val_X, Val_Y, Val_Y_ori, Test_X, Test_Y, Test_Y_ori, X_unlabeled = pickle.load(f)
# Training and test set for GPS segments
prop = 1
random.seed(7)
np.random.seed(7)
tf.set_random_seed(7)
Train_X_Comb = Train_X
index = np.arange(len(Train_X))
np.random.shuffle(index)
Train_X = Train_X[index[:round(prop*len(Train_X))]]
Train_Y = Train_Y[index[:round(prop*len(Train_Y))]]
#Train_X_Comb = np.vstack((Train_X, Train_X_Unlabel))
random.shuffle(Train_X_Comb)



NoClass = 5
Threshold = 200




# Model and Compile
model = Sequential()
activ = 'relu'
model.add(Conv2D(32, (1, 3), strides=(1, 1), padding='same', activation=activ, input_shape=(1, Threshold, 3)))
model.add(Conv2D(32, (1, 3), strides=(1, 1), padding='same', activation=activ))
model.add(MaxPooling2D(pool_size=(1, 2)))

model.add(Conv2D(64, (1, 3), strides=(1, 1), padding='same', activation=activ))
model.add(Conv2D(64, (1, 3), strides=(1, 1), padding='same', activation=activ))
model.add(MaxPooling2D(pool_size=(1, 2)))

model.add(Conv2D(128, (1, 3), strides=(1, 1), padding='same', activation=activ))
model.add(Conv2D(128, (1, 3), strides=(1, 1), padding='same', activation=activ))
model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Dropout(.5))

model.add(Flatten())
A = model.output_shape
model.add(Dense(int(A[1] * 1/4.), activation=activ))
model.add(Dropout(.5))

model.add(Dense(NoClass, activation='softmax'))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

offline_history = model.fit(Train_X, Train_Y, epochs=20, batch_size=100, shuffle=False,
                            validation_data=(Test_X, Test_Y))
hist = offline_history
print('Val_accuracy', hist.history['val_acc'])
print('optimal Epoch: ', np.argmax(hist.history['val_acc']))
# Saving the test and training score for varying number of epochs.
with open('Revised_accuracy_history_largeEpoch_NoSmoothing.pickle', 'wb') as f:
    pickle.dump([hist.epoch, hist.history['acc'], hist.history['val_acc']], f)

A = np.argmax(hist.history['val_acc'])
print('the optimal epoch size: {}, the value of high accuracy {}'.format(hist.epoch[A], np.max(hist.history['val_acc'])))

# Calculating the test accuracy, precision, recall

y_pred = np.argmax(model.predict(Test_X, batch_size=100), axis=1)
print('Test Accuracy %: ', accuracy_score(Test_Y_ori, y_pred))
print('\n')
print('Confusin matrix: ', confusion_matrix(Test_Y_ori, y_pred))
print('\n')
print(classification_report(Test_Y_ori, y_pred, digits=3))
