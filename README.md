# Data visualization exercise 

To explore key features of the data, I decided to extract the sick birds from the dataset, 
cluster them and plot the clusters on a map. The code which does this can be found in `data_viz.py`, and the results of this can be shown in 'sick_birds.html'.
Each bubble represents the center of a cluster and its size is roughly proportional to the spatial distribution of birds in the cluster.

For me, this became an exercise in getting familiar with plotting geographical data in Python using
the plotly package. As such, the resulting map is very crude and needs more work to look nicer. Given more time, 
I would implement a color intensity scale showing the number of birds in a cluster (more intense ==> more birds). 
Right now, the number is seen by hovering over the bubbles.

I would also include some sort of relationship to healthy birds in the same area, to show the proportion of infected
 individuals. There was also an idea of plotting the arrival of sick birds over time, but I did not have time for that.
 
 The task that was most time-consuming was the clustering algorithm, and therefore I could not put as much effort into 
 the visualization - this perhaps indicates that my interest for the data analysis is greater than the visualization.
 
 The clustering works by:
 
 1. Extract the birds who are sick ("result" value 1 or 2)
 2. Assign the geographical coordinates of the first bird to a cluster
 3. For each remaining bird:
    1. Calculate the Euclidian distance from its coordinates to the centroid of each cluster
        1. If the distance to the centroid of a cluster is below a given threshold, add the bird to that cluster
        2. Else, assign the bird to a new cluster
 4. When all birds are assigned to clusters, for each cluster:
    1. Calculate the distance from the center of the cluster to all others
        1. If there is a cluster that is closer than the threshold, merge the current cluster with that
        2. Else, keep the cluster
5. Repeat step 4 until no more cluster merges occur

The clustering can be called through the `threshold_cluster()` function, and the threshold can be set manually. 