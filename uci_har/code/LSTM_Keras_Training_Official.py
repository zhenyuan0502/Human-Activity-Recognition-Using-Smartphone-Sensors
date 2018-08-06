from __future__ import absolute_import, division, print_function
import os
import time
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, GRU, Flatten, LSTM
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

import tensorflow as tf

print(tf.__version__)
print("Loading data")

# UCI 
# https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
features561 = pd.read_csv('data_description/features.txt', header = None, sep='\s+')
train_path = 'uci_har/data/train/'
test_path = 'uci_har/data/test/'
subject_train = pd.read_csv(train_path + 'subject_train.txt', header = None, sep='\s+')

INPUT_SIGNAL_TYPES = [
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

#classification lables
ACTIVITIES = {
    0: 'WALKING',
    1: 'WALKING_UPSTAIRS',
    2: 'WALKING_DOWNSTAIRS',
    3: 'SITTING',
    4: 'STANDING',
    5: 'LAYING',
}

DATADIR = 'uci_har/data/'

SIGNALS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z"
]

def _read_csv(filename):
    return pd.read_csv(filename, delim_whitespace=True, header=None)

def load_signals(subset):
    signals_data = []

    for signal in SIGNALS:
        filename = f'{DATADIR}/{subset}/Inertial Signals/{signal}_{subset}.txt'
        signals_data.append(
            _read_csv(filename).as_matrix()
        ) 

    # Transpose is used to change the dimensionality of the output,
    # aggregating the signals by combination of sample/timestep.
    # Resultant shape is (7352 train/2947 test samples, 128 timesteps, 9 signals)
    return np.transpose(signals_data, (1, 2, 0))

def load_y(subset):
    """
    The objective that we are trying to predict is a integer, from 1 to 6,
    that represents a human activity. We return a binary representation of 
    every sample objective as a 6 bits vector using One Hot Encoding
    (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)
    """
    filename = f'{DATADIR}/{subset}/y_{subset}.txt'
    y = _read_csv(filename)[0]

    return pd.get_dummies(y).as_matrix()

def load_data():
    """
    Obtain the dataset from multiple files.
    Returns: X_train, X_test, y_train, y_test
    """
    X_train, X_test = load_signals('train'), load_signals('test')
    y_train, y_test = load_y('train'), load_y('test')

    return X_train, X_test, y_train, y_test

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

def accuracy_evaluation(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)

checkpoint_path = "uci_har/data/LSTM_CP.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


# for reproducibility
# https://github.com/fchollet/keras/issues/2280
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)

from keras import backend as K
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def _count_classes(y):
    return len(set([tuple(category) for category in y]))

X_train, X_test, Y_train, Y_test = load_data()

# Shuffle the data
seed = abs(hash(time.time()))
if seed > 2**32 - 1:
    seed = int(seed / 2**30)
print("Hash seed: " + str(seed))
np.random.seed(seed)
np.random.shuffle(X_train)
np.random.seed(seed)
np.random.shuffle(X_test)
np.random.seed(seed)
np.random.shuffle(Y_train)
np.random.seed(seed)
np.random.shuffle(Y_test)

timesteps = len(X_train[0])
input_dim = len(X_train[0][0])
n_classes = _count_classes(Y_train)
n_hidden = 32

model = Sequential()
model.add(LSTM(n_hidden, input_shape=(timesteps, input_dim)))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Create checkpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True, save_best_only=True,
                                                 verbose=1, period=5)
# earlystopping_callback = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5,
#                           verbose=1, mode='auto')

earlystopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5,
                          verbose=1, mode='auto')

callbacks_list = [checkpoint_callback]

# Config
numOfEpochs = 50
validation_split = 0.1

start = time.time()
model_info = model.fit(x=X_train, y=Y_train, epochs=numOfEpochs, verbose=1, shuffle=True,
                        batch_size=128, 
                        callbacks=callbacks_list, 
                        # validation_split=validation_split
                        validation_data = (X_test, Y_test),
                        )
end = time.time()

# plot model history
plot_model_history(model_info)

# Save entire model to a HDF5 file
print("Saving keras model...")
model.save('uci_har/data/LSTM_Model.h5')

loss, accuracy = model.evaluate(X_test, Y_test, batch_size=128)
print("On test data, Loss: " + str(loss) + "; Accuracy: " + str(accuracy))
# print("Accuracy on test data is: " + str(accuracy_evaluation(X_test, y_test, model)))
