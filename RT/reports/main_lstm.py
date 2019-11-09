from numpy.random import seed
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy import stats

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Dropout


from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import Normalizer

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


from sklearn.model_selection import GridSearchCV

import pickle
from joblib import dump, load

import sys
from matplotlib import pyplot

seed(42)

features = ['right_mean', 'right_std', 'right_min', 'right_max', 'left_mean', 'left_std', 'left_min', 'left_max', 'mean_pupil']
data = pd.read_csv("features8_w1000.csv", sep=',')

data = data.dropna()

temp_c = list(data.columns)
temp_c.remove('label')

y = data['label']
X = data[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42, shuffle=False)
X_train, y_train = SMOTE().fit_resample(X_train, y_train)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.values.reshape((X_test.values.shape[0], 1, X_test.values.shape[1]))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=20)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
