from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation, Dropout
from keras.utils.np_utils import to_categorical
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

model = Sequential()

model.add(Dense(200, input_dim = 784))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(10))

model.compile(loss = 'categorical_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])

label_train_category = to_categorical(label_train)
label_test_category = to_categorical(label_test)

model.fit(data_train, label_train_category, nb_epoch = 100, batch_size = 100, verbose = 1)

results = model.predict_classes(data_test, verbose = 1)
