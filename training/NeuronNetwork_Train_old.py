import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb
import tensorflow as tf
plt.ion()

# Config
numOfEpoch = 1000

# load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
print('Train Data', train.shape, '\n', train.columns)
print('\nTest Data', test.shape)


# Training area
num_labels = 6
train_x = np.asarray(train.iloc[:, :-2])
train_y = np.asarray(train.iloc[:, 562])
act = np.unique(train_y)
for i in np.arange(num_labels):
    np.put(train_y, np.where(train_y == act[i]), i)
train_y = np.eye(num_labels)[train_y.astype('int')]  # one-hot encoding

test_x = np.asarray(test.iloc[:, :-2])
test_y = np.asarray(test.iloc[:, 562])
for i in np.arange(num_labels):
    np.put(test_y, np.where(test_y == act[i]), i)
test_y = np.eye(num_labels)[test_y.astype('int')]

# shuffle the data
seed = 456
np.random.seed(seed)
np.random.shuffle(train_x)
np.random.seed(seed)
np.random.shuffle(train_y)
np.random.seed(seed)
np.random.shuffle(test_x)
np.random.seed(seed)
np.random.shuffle(test_y)

# place holder variable for x with number of features - 561
x = tf.placeholder('float', [None, 561], name='x')
# place holder variable for y with the number of activities - 6
y = tf.placeholder('float', [None, 6], name='y')
# softmax model


def train_softmax(x):
    W = tf.Variable(tf.zeros([561, 6]), name='weights')
    b = tf.Variable(tf.zeros([6]), name='bias')
    lr = 0.25
    prediction = tf.nn.softmax(tf.matmul(x, W) + b, name='op_predict')
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=prediction, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

    sess = tf.InteractiveSession()
    
    tf.global_variables_initializer().run()
    for epoch in range(numOfEpoch):
        loss = 0
        _, c = sess.run([optimizer, cost], feed_dict={x: train_x, y: train_y})
        loss += c
        if (epoch % 100 == 0 and epoch != 0):
            print('Epoch', epoch, 'completed out of',
                  numOfEpoch, 'Training loss:', loss)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='op_accuracy')

    print('Train set Accuracy:', sess.run(
        accuracy, feed_dict={x: train_x, y: train_y}))
    print('Test set Accuracy:', sess.run(
        accuracy, feed_dict={x: test_x, y: test_y}))


train_softmax(x)

# Neural Network Classifier
n_nodes_input = 561  # number of input features
n_nodes_hl = 30     # number of units in hidden layer
n_classes = 6       # number of activities
x = tf.placeholder('float', [None, 561])
y = tf.placeholder('float')

def neural_network_model(data):
    # define weights and biases for all each layer
    hidden_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_input, n_nodes_hl], stddev=0.3)),
                    'biases': tf.Variable(tf.constant(0.1, shape=[n_nodes_hl]))}
    output_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl, n_classes], stddev=0.3)),
                    'biases': tf.Variable(tf.constant(0.1, shape=[n_classes]))}
    # feed forward and activations
    l1 = tf.add(tf.matmul(data, hidden_layer['weights']), hidden_layer['biases'])
    l1 = tf.nn.sigmoid(l1)
    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']
    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for epoch in range(numOfEpoch):
        loss = 0
        _, c = sess.run([optimizer, cost], feed_dict={x: train_x, y: train_y})
        saver.save(sess, "/tmp/model_neural_network.chkp")
        loss += c
        if (epoch % 100 == 0 and epoch != 0):
            print('Epoch', epoch, 'completed out of',
                  numOfEpoch, 'Training loss:', loss)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='op_accuracy')

    print('Train set Accuracy:', sess.run(
        accuracy, feed_dict={x: train_x, y: train_y}))
    print('Test set Accuracy:', sess.run(
        accuracy, feed_dict={x: test_x, y: test_y}))

train_neural_network(x)
