import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Perceptron
import time
import plot_decision_region

df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

f1 = plt.figure(1)
f2 = plt.figure(2)
plt.figure(f1.number)
plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color = 'blue', marker = 'x', label = 'versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc = 'upper left')

plt.figure(f2.number)
ppn = Perceptron.Perceptron(eta = 0.1, n_iter = 10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker = 'o')
plt.xlabel('Epoches')
plt.ylabel('Number of misclasifications')

plt.figure(3)

plot_decision_region.plot_decision_region(X, y, classifier = ppn)
plt.show()
