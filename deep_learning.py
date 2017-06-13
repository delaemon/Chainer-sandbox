#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import time
from chainer import Chain, Variable, optimizers
import chainer.links as L
import chainer.functions as F
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals import joblib
import numpy as np

mnist = fetch_mldata("MNIST original", data_home=".")
data = np.asarray(mnist.data, np.float64)
dataset = train_test_split(data, mnist.target, test_size=0.2)
joblib.dump(dataset, "mnist")

data_train, data_test, label_train, label_test = joblib.load("mnist")
data_train = np.asarray(data_train, np.float32)
data_test = np.asarray(data_test, np.float32)
label_train = np.asarray(label_train, np.int32)
label_test = np.asarray(label_test, np.int32)

class MyNetwork(Chain):
  def __init__(self):
    super(MyNetwork, self).__init__(
      l1=L.Linear(784, 200),
      l2=L.Linear(200, 100),
      l3=L.Linear(100, 10))

  def __call__(self, x):
    h1 = F.dropout(F.relu(self.l1(x)))
    h2 = F.dropout(F.relu(self.l2(h1)))
    p = self.l3(h2)
    return p

network = MyNetwork()
model = L.Classifier(network)
model.compute_accuracy = True

optimizer = optimizers.Adam()
optimizer.setup(model)

n_epoch = 100
batchsize = 20
N = len(data_train)
losses = []

start = time.time()
for epoch in range(n_epoch):
  print('epoch: %d' % (epoch + 1))
  perm = np.random.permutation(N)
  sum_accuracy = 0
  sum_loss = 0
  for i in range(0, N, batchsize):
    x_batch = data_train[perm[i:i+batchsize]]
    t_batch = label_train[perm[i:i+batchsize]]

    model.zerograds()

    x = Variable(x_batch)
    t = Variable(t_batch)
    loss = model(x, t)

    loss.backward()

    accuracy = model.accuracy

    optimizer.update()

    sum_loss += float(loss.data) * batchsize
    sum_accuracy += float(accuracy.data) * batchsize

  losses.append(sum_loss / N)
  print("loss: %f, accuracy: %f" % (sum_loss / N, sum_accuracy / N))

training_time = time.time() - start
joblib.dump((model, training_time, losses), "classifiers/"+"nn_cpu")

start = time.time()
x_test = Variable(data_test)
result_scores = network(x_test).data
predict_time = time.time() - start
results = np.argmax(result_scores, axis = 1)

score = accuracy_score(label_test, results)
print(training_time, predict_time)
print(score)
cmatrix = cunfusion_matrix(label_test,results)
print(cmatrix)
joblib.dump((training_time, predict_time, score, cmatrix), "results/"+"nn_cpu")
