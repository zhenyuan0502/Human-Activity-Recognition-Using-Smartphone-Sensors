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
new_model = tf.keras.models.load_model('kaggle_har/data/ANN_Model.h5')
plot_model(new_model, to_file='kaggle_har/results/model_visualization.png')

train = pd.read_csv('kaggle_har/data/train.csv')
test = pd.read_csv('kaggle_har/data/test.csv')

#Feature matrix
train_features = train.iloc[:, :561].as_matrix()
test_results = test.iloc[:, 562:].as_matrix()

test_features = test.iloc[:, :561].as_matrix()
test_resultss = np.zeros((len(test_results), 6))

for k in range(0, len(test_results)):
    if test_results[k] == 'STANDING':
        test_resultss[k][0] = 1
    elif test_results[k] == 'WALKING':
        test_resultss[k][1] = 1
    elif test_results[k] == 'WALKING_UPSTAIRS':
        test_resultss[k][2] = 1
    elif test_results[k] == 'WALKING_DOWNSTAIRS':
        test_resultss[k][3] = 1
    elif test_results[k] == 'SITTING':
        test_resultss[k][4] = 1
    else:
        test_resultss[k][5] = 1

def check_activity(index):
    if index == 0:
        return 'STANDING'
    elif index == 1:
        return 'WALKING'
    elif index == 2:
        return 'WALKING_UPSTAIRS'
    elif index == 3:
        return 'WALKING_DOWNSTAIRS'
    elif index == 4:
        return 'SITTING'
    else:
        return 'LAYING'

print(new_model.predict_classes(test_features))
