import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
import mglearn

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
plt.show()