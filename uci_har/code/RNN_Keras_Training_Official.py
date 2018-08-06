from __future__ import absolute_import, division, print_function
import os
import time
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, GRU, Flatten
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
np.random.shuffle(y_train)
np.random.seed(seed)
np.random.shuffle(y_test)

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

checkpoint_path = "uci_har/data/RNN_CP.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model = Sequential()

# model.add(Dense(64, activation='relu', input_shape=(None, X_train.shape[-1])))

model.add(GRU(32, input_shape=(None, X_train.shape[-1])))
model.add(Dense(1))

# Define optimizer
# sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy',
#               optimizer=sgd,
#               metrics=['accuracy'])

model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])


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
model_info = model.fit(x=X_train, y=y_train, epochs=numOfEpochs, verbose=1, shuffle=True,
                        batch_size=128, 
                        callbacks=callbacks_list, 
                        # validation_split=validation_split
                        validation_data = (X_test, y_test),
                        )
end = time.time()

# plot model history
plot_model_history(model_info)

# Save entire model to a HDF5 file
print("Saving keras model...")
model.save('uci_har/data/RNN_Model.h5')

loss, accuracy = model.evaluate(X_test, y_test, batch_size=128)
print("On test data, Loss: " + str(loss) + "; Accuracy: " + str(accuracy))
# print("Accuracy on test data is: " + str(accuracy_evaluation(X_test, y_test, model)))
