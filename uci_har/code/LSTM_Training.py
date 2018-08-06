import time
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

#importinig required libraries for LSTM
import tensorflow as tf  # Version 1.0.0
from sklearn import metrics
import os

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

#input data
training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])  # 9 input parameters per timestep

#LSTM Neural Network's internal structure
n_hidden = 32 # Hidden layer num of features
n_classes = 6 # Total classes (should go up, or should go down)

#training 
learning_rate = 0.0025
lambda_loss_amount = 0.0015
training_iters = training_data_count * 300  # Loop 300 times on the dataset
batch_size = 1500
display_iter = 30000  # To show test set accuracy during training

def LSTM_RNN(_X, _weights, _biases):
    #input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    #Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) 
    #new shape: (n_steps*batch_size, n_input)
    
    #Linear activation
    _X = tf.nn.tanh(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    #Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0) 
    #new shape: n_steps * (batch_size, n_hidden)

    #Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    #Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    #Get last time step's output feature for a "many to one" style classifier, 
    #as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]
    
    #Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']

#function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
def extract_batch_size(_train, step, batch_size):
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 

    return batch_s

#function to encode output labels from number indexes
def one_hot(y_):
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

#graph input/output
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


#graph weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden]),name='input'), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = LSTM_RNN(x, weights, biases)
pred_softmax = tf.nn.softmax(pred, name="y_")

#add l2 to prevent over-fitting
l2 = lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() ) 
# L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) +l2 # Softmax loss
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

correct_pred = tf.equal(tf.argmax(pred_softmax,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#to keep track of training's performance
saver = tf.train.Saver()
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []
history = dict(train_loss=[], 
                     train_acc=[], 
                     test_loss=[], 
                     test_acc=[])

#launch the graph
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

#perform training steps with "batch_size" amount of example data at each loop
step = 1
while step * batch_size <= training_iters:
    batch_xs = extract_batch_size(X_train, step, batch_size)
    batch_ys = one_hot(extract_batch_size(y_train, step, batch_size))

    #fit training using batch data
    _, loss, acc = sess.run(
        [optimizer, cost, accuracy],
        feed_dict={
            x: batch_xs, 
            y: batch_ys
        }
    )
    train_losses.append(loss)
    train_accuracies.append(acc)
    history['train_loss'].append(loss)
    history['train_acc'].append(acc)
    
    #evaluate network only at some steps for faster training: 
    if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
        
        #to not spam console, show training accuracy/loss in this "if"
        print("Training iter #" + str(step*batch_size) + \
              ":   Batch Loss = " + "{:.6f}".format(loss) + \
              ", Accuracy = {}".format(acc))
        
        #evaluation on the test set (no learning made here - just evaluation for diagnosis)
        loss, acc = sess.run(
            [cost, accuracy], 
            feed_dict={
                x: X_test,
                y: one_hot(y_test)
            }
        )
        test_losses.append(loss)
        test_accuracies.append(acc)
        history['test_loss'].append(loss)
        history['test_acc'].append(acc)
        print("PERFORMANCE ON TEST SET: " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc))

    step += 1

print("Optimization Finished!")
