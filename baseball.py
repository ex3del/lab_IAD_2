import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# 'Name', 'Team', 'Position', 'Height(inches)', 'Weight(pounds)', 'Age'
df = pd.read_csv('baseball_players.csv', delimiter=',')

# Позиции игроков:
# 'Relief_Pitcher', 'Shortstop', 'Second_Baseman', 'Third_Baseman', 'First_Baseman',
# 'Catcher', 'Starting_Pitcher', 'Designated_Hitter', 'Outfielder'
df_work = df.sample(n=200, random_state=100)

positions = ['Relief_Pitcher', 'Shortstop', 'Second_Baseman', 'Third_Baseman',
             'First_Baseman', 'Catcher', 'Starting_Pitcher', 'Designated_Hitter', 'Outfielder']
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'g']
clr = dict(zip(positions, colors))

def box_plots():
    for caharateristic in df:
        print(df[caharateristic].describe(), end='\n*****************\n')
    sb.boxplot(data=df)
    plt.show()


def heat_map(dat=df):
    sb.heatmap(dat.corr(), annot=True, cmap='coolwarm', linewidths=3, linecolor='black')
    plt.show()


def threeD_plot(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in positions:
        tab = df_work.loc[df_work.Position == i, :]
        print(tab.head())
        ax.scatter(tab[x], tab[y], tab[z], label=i)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title('Характеристики игроков')
    ax.legend()
    plt.show()


def ierarchy():
    metd = ['single', 'complete', 'average', 'centroid', 'ward']
    metr = ['euclidean', 'cityblock', 'chebyshev', 'cosine']
    for i in metd:
        for j in metr:
            igroki = linkage(df_work.drop(columns=['Name', 'Team', 'Position']), method=i, metric=j)
            dendrogram(igroki)
            dendrogram(igroki, leaf_rotation=90, leaf_font_size=6)
            plt.title(f'Mtd: {i}, mtr:{j}')
            plt.show()


def pca(tabl):
    features = ['Height(inches)', 'Weight(pounds)', 'Age']
    tabl = tabl.replace({'Position': {0: 'Relief_Pitcher', 1: 'Shortstop', 2: 'Second_Baseman', 3: 'Third_Baseman',
                                      4: 'First_Baseman', 5:'Catcher', 6: 'Starting_Pitcher', 7: 'Designated_Hitter',
                                      8: 'Outfielder' }})
    x = tabl.loc[:, features].values
    y = tabl.loc[:, []].values
    x = StandardScaler().fit_transform(x) # pd.DataFrame(data=x, columns=features).head())
    pca = PCA(n_components=2)
    prcp_compon = pca.fit_transform(x)
    prcp_df = pd.DataFrame(data=prcp_compon, columns=['PC1', 'PC2'])
    final_df = pd.concat([prcp_df, tabl['Position']], axis=1)
    print(pca.explained_variance_ratio_)
    def pca_graph():
        plt.scatter(final_df['PC1'], final_df['PC2'])
        plt.grid()
        plt.legend()
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('2 Component PCA')
        plt.show()


def tsne(table):
    table = table.drop(columns=['Name', 'Team', 'Position'])
    features = ['Height(inches)', 'Weight(pounds)', 'Age']
    X = table.loc[:, features].values
    igroki = TSNE(n_components=2, perplexity=5, early_exaggeration=15, n_iter=8000, random_state=100)
    X_igroki = pd.DataFrame(igroki.fit_transform(X))
    print('Kullback-Leibler divergence after optimization: ', igroki.kl_divergence_)
    print('No. of iterations: ', igroki.n_iter_)
    plt.scatter(X_igroki[0], X_igroki[1])
    plt.grid()
    plt.legend()
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('t-SNE')
    plt.show()


def k_means(table):
    table = table.drop(columns=['Name', 'Team', 'Position'])
    model = KMeans(n_clusters=9, random_state=100)
    model.fit(table)
    all_predict = model.predict(table)
    def histog():
        plt.hist(all_predict, bins=16)
        plt.xlabel('Номер класса')
        plt.ylabel('Предполагаемое кол-во эл-ов')
        plt.show()
    return all_predict


def six_punk(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    table = df_work.drop(columns=['Name', 'Team'])
    table['Position'] = k_means(df_work)
    for i in range(9):
        tab = table.loc[table.Position == i, :]
        ax.scatter(tab[x], tab[y], tab[z])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    plt.title('Предсказанное распределение пункта 6')
    ax.legend()
    plt.show()

six_punk('Height(inches)', 'Weight(pounds)', 'Age')