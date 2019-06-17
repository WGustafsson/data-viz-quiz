import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
import plotly
import plotly.plotly as py
import plotly.graph_objs as go


def centroid(cluster):
    cluster = np.asarray(cluster)

    c = []

    for i in range(cluster.shape[1]):
        c.append(np.mean([x[i] for x in cluster]))

    return tuple(c)


def euclid_distance(a, b):
    assert(len(a) == len(b))
    squares = []
    for i in range(len(a)):
        squares.append((a[i] - b[i])**2)
    return np.sqrt(np.sum(squares))


def threshold_cluster(data, threshold):

    # assign first point as seed of first cluster
    clusters = [[data[0]]]
    centroids = [data[0]]

    for i in range(1, len(data)):
        for j in range(len(clusters)):
            # if current data point is closer than threshold to a cluster, assign the point to that cluster
            if euclid_distance(centroids[j], data[i]) < threshold:
                clusters[j].append(data[i])
                centroids[j] = centroid(clusters[j]) # recalculate centroid of this cluster
                break
        # if not within threshold of any cluster, assign data point as seed for new cluster
        else:
            clusters.append([data[i]])
            centroids.append(centroid([data[i]]))

    clusters_copy = clusters.copy()
    centroids_copy = centroids.copy()
    for c1 in range(len(clusters)):
        for c2 in range(len(clusters_copy)):
            if centroids[c1] != centroids_copy[c2]:
                if euclid_distance(centroids[c1], centroids_copy[c2]) < threshold:


data_path = os.path.dirname(os.path.realpath(__file__)) + '/data/Avian_Influenza.csv'

df = pd.read_csv(data_path, sep=';')

print(df.shape)
print(df.result.unique())

sick_df = df.loc[df['result'] >= 1]
healthy_df = df.loc[df['result'] == 0]

print(sick_df.shape)
print(healthy_df.shape)

print(sick_df)

sick_coordinates = list(zip(sick_df['coords.x1'], sick_df['coords.x2']))

print(sick_coordinates)

data = [go.Scattergeo(locationmode='country names', lon=sick_df['coords.x1'], lat=sick_df['coords.x2'],
                      text=sick_df['species'], mode='markers')]

layout = dict(title='Sick birds', geo=dict(scope='europe', showland=True, showlakes=True, showcountries=True,
                                           resolution=50, projection=dict(type='mercator')))

fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig)
