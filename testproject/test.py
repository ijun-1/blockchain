#
import numpy as np
import scipy as sp
import pandas as pd
from pandas import Series,DataFrame
#可視化ライブラリ
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
#%matplotlib inline
#小数点
#%precision 3

from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from pandas.plotting import scatter_matrix

np.random.seed(0)



data = load_iris()
X = pd.DataFrame(data.data,columns=data.feature_names)
y = pd.DataFrame(data.target,columns=["Species"])
df = pd.concat([X,y],axis=1)
print(df.head())



