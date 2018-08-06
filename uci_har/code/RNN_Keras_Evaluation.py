from __future__ import absolute_import, division, print_function
import os

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
from keras.utils import plot_model

import tensorflow as tf
from Plot_Confusion_Matrix import plot_confusion_matrix

print(tf.__version__)
print("Loading model from file")
new_model = tf.keras.models.load_model('uci_har/data/RNN_Model.h5')
new_model.summary()
plot_model(new_model, to_file='uci_har/results/model_visualization.png')

train_path = 'uci_har/data/train/'
test_path = 'uci_har/data/test/'

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


X_train_signals_paths = [
    train_path + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    test_path + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)

y_train_path = train_path + "y_train.txt"
y_test_path = test_path + "y_test.txt"

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

loss, acc = new_model.evaluate(X_test, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

test_actual_labels = y_test
test_pred_labels = new_model.predict_classes(X_test)

# Confusion matrix for testing data
class_names = ['STANDING', 'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'LAYING'] 
plot_confusion_matrix(cm=confusion_matrix(test_actual_labels, test_pred_labels), normalize = False,
                        target_names = class_names, title='Confusion matrix, without normalization')

plot_confusion_matrix(cm=confusion_matrix(test_actual_labels, test_pred_labels), normalize = True,
                        target_names = class_names, title='Confusion matrix, with normalization')