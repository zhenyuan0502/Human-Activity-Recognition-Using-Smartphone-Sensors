from __future__ import absolute_import, division, print_function
import os
import time
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, GRU
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
LABELS = [
    "WALKING", 
    "WALKING_UPSTAIRS", 
    "WALKING_DOWNSTAIRS", 
    "SITTING", 
    "STANDING", 
    "LAYING"
]

#Load X (train + test)
def load_X(X_signals_paths):
    X_signals = []
    
    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()
    
    return np.transpose(np.array(X_signals), (1, 2, 0))

X_train_signals_paths = [
    train_path + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    test_path + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)

#Load y (train + test)
def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]], 
        dtype=np.int32
    )
    file.close()
    
    #Substract 1 to each output class for friendly 0-based indexing 
    return y_ - 1

y_train_path = train_path + "y_train.txt"
y_test_path = test_path + "y_test.txt"

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

#Converting 3D dataframe to 2D to implement standard neural network algorithms
#xtrain_df = pd.DataFrame(X_train.reshape(len(X_train),1152))
xtrain_df = pd.DataFrame(X_train.reshape(len(X_train),768))
ytrain_df = pd.DataFrame(y_train.reshape(len(y_train),1))
#xtest_df = pd.DataFrame(X_test.reshape(len(X_test),1152))
xtest_df = pd.DataFrame(X_test.reshape(len(X_test),768))
ytest_df = pd.DataFrame(y_test.reshape(len(y_test),1))

#Plotting the frequency of training data by activity type
# plt.show(ytrain_df[0].value_counts().sort_index().plot(kind='bar', title='Frequency of Training examples by Activity Type'))

#Plotting the frequency of training data by user
# plt.show(subject_train[0].value_counts().sort_index().plot(kind='bar', title='Frequency of Training examples by user'))

#adding fetaures as headers to xtrain
#xtrain_df.columns=feature561_df[1]
ytrain_df.columns=['Activity']
subject_train.columns=['User']

#concatinating activity and user columns to each observation
train_df=pd.concat([xtrain_df,ytrain_df,subject_train], axis=1)

#User-wise Activity frequency
pd.crosstab(train_df.User, train_df.Activity)


# Implementing Recurrent Neural Networks with keras
model = Sequential()
model.add(GRU(32, input_shape=(None, X_train.shape[-1])))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

history = model.fit(x=X_train,
                    y=y_train,
                    batch_size=128,
                    epochs=50,
                    verbose=1,
                    callbacks=None,
                    validation_data = (X_test, y_test),
                    shuffle=True,
                   )
history_dict = history.history
history_dict.keys()

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

loss = history.history['acc']
val_loss = history.history['val_acc']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'g', label='Training acc')
plt.plot(epochs, val_loss, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.legend()

plt.show()
