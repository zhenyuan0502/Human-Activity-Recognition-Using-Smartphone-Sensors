import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb
plt.ion()

# # load data
# train = pd.read_csv("data/train.csv")
# test = pd.read_csv("data/test.csv")
# print('Train Data', train.shape, '\n', train.columns)
# print('\nTest Data', test.shape)

# # Exploratory Analysis
# print('Train labels', train['Activity'].unique(), 
# '\nTest Labels', test['Activity'].unique())
# pd.crosstab(train.subject, train.Activity)

# sub15 = train.loc[train['subject']==15]

# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=6).fit(sub15.iloc[:,:-2])
# clust = pd.crosstab(kmeans.labels_, sub15['Activity'])
# print(clust)
# print(kmeans.cluster_centers_.shape)


import tensorflow as tf
v1 = tf.get_variable("v1", shape=[3])

saver = tf.train.Saver()

with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())

