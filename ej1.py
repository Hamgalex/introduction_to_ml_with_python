import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# breast cancer
cancer=load_breast_cancer()
print("Cancer.keys():\n{}".format(cancer.keys())) # no solo trae los puros datos sino también trae info adicional como descrp, target, etc

print("Shape of cancer data: {}".format(cancer.data.shape)) # hay 569 registros y 30 columnas

df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
print(df.head())

# boston
boston = load_boston()
print("Data shape: {}".format(boston.data.shape))

X, y = mglearn.datasets.load_extended_boston() # joblib==1.1.0
print("X.shape: {}".format(X.shape))

mglearn.plots.plot_knn_classification(n_neighbors=4) # se usa el forge
# plt.show()

# knn en forge con 3 vecinos
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print("Test set predictions: {}".format(clf.predict(X_test))) # se ve que tan bien se entrenó
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test))) # se ve el score
