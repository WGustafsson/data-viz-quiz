import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly
import plotly.plotly as py
import plotly.graph_objs as go


def euclid_distance(a, b):
    assert(len(a) == len(b))
    squares = []
    for i in range(len(a)):
        squares.append((a[i] - b[i])**2)
    return np.sqrt(np.sum(squares))


def centroid(cluster):
    cluster = np.asarray(cluster)

    c = []

    for i in range(cluster.shape[1]):
        c.append(np.mean([x[i] for x in cluster]))

    return tuple(c)


def threshold_cluster(data, threshold):

    # assign first point as seed of first cluster
    clusters = {(data[0],): data[0]} # cluster: centroid

    for i in range(1, len(data)):
        for c, ce in clusters.items():
            # if current data point is closer than threshold to a cluster, assign the point to that cluster
            if euclid_distance(ce, data[i]) < threshold:
                del clusters[c]
                c = list(c)
                c.append(data[i])
                cent = centroid(c) # recalculate centroid of this cluster
                c = tuple(c)
                clusters[c] = cent
                break
        # if not within threshold of any cluster, assign data point as seed for new cluster
        else:
            clusters[(data[i],)] = data[i]

    change = True
    while change:
        clusters_copy = clusters.copy()
        for c1, ce1 in clusters.items():
            for c2, ce2 in clusters_copy.items():
                if ce1 != ce2:
                    if euclid_distance(ce1, ce2) < threshold:
                        change = True
                        del clusters_copy[c2]
                        c3 = list(c1) + list(c2)
                        ce3 = centroid(c3)
                        clusters_copy[tuple(c3)] = ce3
                        break
            else:
                change = False
        clusters = clusters_copy

    ret = {}
    i = 1
    for k, v in clusters.items():
        ret[i] = (k, v)
        i += 1

    return ret

data_path = os.path.dirname(os.path.realpath(__file__)) + '/data/Avian_Influenza.csv'

df = pd.read_csv(data_path, sep=';')

sick_df = df.loc[df['result'] >= 1]
healthy_df = df.loc[df['result'] == 0]

sick_coordinates = list(zip(sick_df['coords.x1'], sick_df['coords.x2']))

print(len(sick_coordinates))

clusters = threshold_cluster(sick_coordinates, 2)

print(len(clusters))

sample_clusters = []
centroid_x1 = []
centroid_x2 = []

for index, row in sick_df.iterrows():
    for k, v in clusters.items():
        coords = (row['coords.x1'], row['coords.x2'])
        if  coords in v[0]:
            sample_clusters.append(k)
            centroid_x1.append(v[1][0])
            centroid_x2.append(v[1][1])
            break

sick_df['cluster'] = sample_clusters
sick_df['centroid_x1'] = centroid_x1
sick_df['centroid_x2'] = centroid_x2

n = len(sick_df.index)

cluster_group = sick_df.groupby('cluster')

cluster_counts = cluster_group.size()
cluster_longcentroid = cluster_group['centroid_x1'].first()
cluster_latcentroid = cluster_group['centroid_x2'].first()
cluster_longmin = cluster_group['coords.x1'].min()
cluster_longmax = cluster_group['coords.x1'].max()
cluster_latmin = cluster_group['coords.x2'].min()
cluster_latmax = cluster_group['coords.x2'].max()

cluster_df = pd.concat([cluster_counts, cluster_longcentroid, cluster_latcentroid, cluster_longmin, cluster_longmax, cluster_latmin, cluster_latmax], axis=1)

cluster_df.columns = ['count', 'long_centroid', 'lat_centroid', 'long_min', 'long_max', 'lat_min', 'lat_max']

marker_sizes = [max(row['long_max'] - row['long_centroid'], 0.1)*100 for _, row in cluster_df.iterrows()]

cluster_df['marker_size'] = marker_sizes

print(cluster_df)

data = [go.Scattergeo(lon=cluster_df['long_centroid'], lat=cluster_df['lat_centroid'],
                      text=list("%d birds" % (x) for x in cluster_df['count']), mode='markers', marker=go.scattergeo.
                      Marker(size=cluster_df['marker_size'], color = 'red'))]

layout = dict(autosize=True, title='Sick bird clusters', geo=dict(scope='europe', showland=True, showlakes=True, showcountries=True,
                                           resolution=50, projection=dict(type='mercator')))

fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='sick_birds.html')
