import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import pingouin as pg
from sklearn.cluster import KMeans

iris = load_iris()

#  'sepal length (cm)' , 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
iris_pd = pd.DataFrame(data=np.c_[iris['data']], columns=iris['feature_names'])
# iris_pd = iris_pd.drop(columns=['sepal width (cm)'])

def threeD_plot(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(iris_pd[x], iris_pd[y], iris_pd[z], label='Характеристики цветков')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.legend()
    plt.show()

threeD_plot('sepal length (cm)', 'petal length (cm)', 'petal width (cm)')
