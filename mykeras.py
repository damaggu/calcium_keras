from __future__ import print_function

# fix random seed for reproducibiliy
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, LSTM, Activation
from keras import backend as K
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

print(os.getcwd())

batch_size = 128
epochs = 12

window_size = 5000

dataset = np.load('../spikefinder/fuel_data/' + 'v2_train_calcium.npy')
labels = np.load('../spikefinder/fuel_data/' + 'v2_train_spikes.npy')


print(labels.shape)

# 800000
# 980200

# testing
x_train = dataset[0:800000,:]
y_train = labels[0:800000,:]

x_test = dataset[800001:980200,:]
y_test = labels[800001:980200,:]


#training
# x_train = dataset[0:2000,:]
# y_train = labels[0:2000,:]
#
# x_test = dataset[2001:3000,:]
# y_test = labels[2001:3000,:]


model = Sequential()
# model.add(Conv1D(input_shape=(8000,500),kernel_size=200,strides=1,filters=20))
model.add(Dense(200, activation='relu', input_dim=500))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(LSTM())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])


history = model.fit(x_train, y_train, epochs=5,
                    batch_size=32,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test, batch_size=128)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print(score)

# start prediction

mymetric = 0
out_of = 0
predictions = []
real_vals = []
for idx, entry in enumerate(dataset[0:2000,:]):
    entry = [entry]
    entry = np.asarray(entry)
    prediction = model.predict(entry, batch_size=1)
    predictions.append(prediction)
    real_vals.append(labels[idx])
    if(labels[idx] == 1.0):
        print('model: ' + str(prediction) + '  |   real: ' + str(labels[idx]) + "\n")
        out_of += 1
        if(prediction >= 0.9):
            mymetric += 1
            print('schmo')

print('summe:  '  + str(mymetric))
print('out_of'    + str(out_of))


# predictions = np.asarray(predictions)
# real_vals = np.asarray(real_vals)

# print(zip(predictions, real_vals))
# plt.plot(predictions.squeeze())
# plt.show()

# get layer specific output
# # with a Sequential model
# example = dataset[0]
# get_output = K.function([model.layers[0].input],
#                                   [model.layers[4].output])
# layer_output = get_output([example])[0]
# print(layer_output)

# plt.plot(real_vals)
# plt.show()

# predictions = []
# real_vals = []
# for idx, entry in enumerate(dataset[0:100,:]):
#     prediction = model.predict(entry)
#     predictions.append(prediction)
#     real_vals.append(labels[idx])

# model = Sequential()
# model.add(Conv1D(32, kernel_size=3,
#                  activation='relu',
#                  input_shape=(None, 500)))
# model.add(Conv1D(64, kernel_size=3, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.25))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss=keras.losses.mean_squared_error,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
