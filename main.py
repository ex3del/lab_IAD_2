import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import pingouin as pg
from scipy.cluster.hierarchy import linkage, dendrogram, distance

iris = load_iris()

#  'sepal length (cm)' ,  'petal length (cm)', 'petal width (cm)', 'sepal width (cm)'
iris_pd = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])


def threeD_plot(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(iris_pd[x], iris_pd[y], iris_pd[z], c=iris_pd.target, label='Характеристики цветков')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.legend()
    plt.show()


def ierarchy():
    # 'single', 'complete', 'average', 'centroid', 'ward'
    # 'euclidian', 'cityblock', 'chebyshev', 'cosine'
    cvetochki = linkage(iris_pd.drop(columns=['sepal width (cm)', 'target']), method='ward', metric='cosine')
    dendrogram(cvetochki)
    dendrogram(cvetochki, leaf_rotation=90, leaf_font_size=6)
    plt.show()
ierarchy()



# threeD_plot('sepal length (cm)', 'petal length (cm)', 'petal width (cm)')
