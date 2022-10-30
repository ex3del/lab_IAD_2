import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import pingouin as pg
from scipy.cluster.hierarchy import linkage, dendrogram, distance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



iris = load_iris()
# 0 - iris_setosa, 1- iris_versicolor, 2-iris_virginica
# 'sepal length (cm)' ,  'petal length (cm)', 'petal width (cm)', 'sepal width (cm)'
iris_pd = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])


# threeD_plot('sepal length (cm)', 'petal length (cm)', 'petal width (cm)')
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
    # 'euclidean', 'cityblock', 'chebyshev', 'cosine'
    cvetochki = linkage(iris_pd.drop(columns=['target']), method='complete', metric='euclidean')
    dendrogram(cvetochki)
    dendrogram(cvetochki, leaf_rotation=90, leaf_font_size=6)
    plt.show()


features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
iris_pd.loc[[iris_pd['target'] == 0.0],'target'] = 'iris_setosa'
x = iris_pd.loc[:, features].values
y = iris_pd.loc[:, ['target']].values
x = StandardScaler().fit_transform(x) # pd.DataFrame(data=x, columns=features).head())
pca = PCA(n_components=2)
prcp_compon = pca.fit_transform(x)
prcp_df = pd.DataFrame(data = prcp_compon, columns=['PC1', 'PC2'])
final_df = pd.concat([prcp_df, iris_pd['target']], axis=1)
print(iris_pd.head())
