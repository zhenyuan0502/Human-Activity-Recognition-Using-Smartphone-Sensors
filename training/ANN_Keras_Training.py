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
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

import tensorflow as tf

print(tf.__version__)
print("Loading data")

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

#Feature matrix
train_features = train.iloc[:, :561].as_matrix()
test_features = test.iloc[:, :561].as_matrix()
train_results = train.iloc[:, 562:].as_matrix()
test_results = test.iloc[:, 562:].as_matrix()
train_resultss = np.zeros((len(train_results), 6))
test_resultss = np.zeros((len(test_results), 6))

# Shuffle the data
seed = abs(hash(time.time()))
if seed > 2**32 - 1:
    seed = int(seed / 2**30)
print("Hash seed: " + str(seed))
np.random.seed(seed)
np.random.shuffle(train_features)
np.random.seed(seed)
np.random.shuffle(train_results)
np.random.seed(seed)
np.random.shuffle(test_features)
np.random.seed(seed)
np.random.shuffle(test_results)

for k in range(0, len(train_results)):
    if train_results[k] == 'STANDING':
        train_resultss[k][0] = 1
    elif train_results[k] == 'WALKING':
        train_resultss[k][1] = 1
    elif train_results[k] == 'WALKING_UPSTAIRS':
        train_resultss[k][2] = 1
    elif train_results[k] == 'WALKING_DOWNSTAIRS':
        train_resultss[k][3] = 1
    elif train_results[k] == 'SITTING':
        train_resultss[k][4] = 1
    else:
        train_resultss[k][5] = 1

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

checkpoint_path = "data/ANN_CP.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=561))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(6, activation='softmax'))


# Define optimizer
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# Create checkpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True, save_best_only=True,
                                                 verbose=1, period=5)
# earlystopping_callback = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5,
#                           verbose=1, mode='auto')

earlystopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5,
                          verbose=1, mode='auto')

callbacks_list = [checkpoint_callback, earlystopping_callback]

# Config
numOfEpochs = 10000
validation_split = 0.1

start = time.time()
model_info = model.fit(train_features, train_resultss, epochs=numOfEpochs, 
                        batch_size=128, callbacks=callbacks_list, validation_split=validation_split)
end = time.time()

# plot model history
plot_model_history(model_info)

# Save entire model to a HDF5 file
print("Saving model...")
model.save('data/ANN_Model.h5')


loss, accuracy = model.evaluate(test_features, test_resultss, batch_size=128)
print("On test data, Loss: " + str(loss) + "; Accuracy: " + str(accuracy))
print("Accuracy on test data is: " + str(accuracy_evaluation(test_features, test_resultss, model)))
