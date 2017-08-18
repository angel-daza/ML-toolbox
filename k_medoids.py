import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn.datasets as sk_data
from sklearn.metrics.pairwise import euclidean_distances

# Get top-k values instead of argmax which only gets top value
# max_rows = np.argpartition(np.mean(distance_matrix, axis=1), -k)[-k:]

def cluster(distances, k=3):
    m = distances.shape[0] # number of points

    # Pick k random medoids.
    curr_medoids = np.array([-1]*k)
    while not len(np.unique(curr_medoids)) == k:
        curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
    old_medoids = np.array([-1]*k) # Doesn't matter what we initialize these to.
    new_medoids = np.array([-1]*k)

    curr_medoids = compute_initial_medoids(distances,k)

    # Until the medoids stop updating, do the following:
    while not ((old_medoids == curr_medoids).all()):
        # Assign each point to cluster with closest medoid.
        clusters = assign_points_to_clusters(curr_medoids, distances)

        # Update cluster medoids to be lowest cost point.
        for curr_medoid in curr_medoids:
            cluster = np.where(clusters == curr_medoid)[0]
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)

        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]

    return clusters, curr_medoids

def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:,medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters

def _look_for_new_best(new_elem,max_cols,sum_cols):
    candidates = []
    for i,elem in enumerate(sum_cols):
        if i not in max_cols and i != new_elem:
            candidates.append(elem)
        else:
            candidates.append(0)
    return np.argmax(np.array(candidates))

def compute_initial_medoids(distance_matrix,k=3):
    max_col = np.argmax(np.sum(distance_matrix, axis=1))
    max_cols = [max_col]
    if k==1: return max_cols
    max_cols.append(np.argmax(distance_matrix[max_col]))
    if k==2: return max_cols
    sum_cols = np.sum(distance_matrix[max_cols],axis=0)
    for i in range(k-2):
        new_col = np.argmax(sum_cols)
        if new_col in max_cols: new_col = _look_for_new_best(new_col,max_cols,sum_cols)
        max_cols.append(new_col)
        sum_cols = np.sum(np.array([sum_cols, distance_matrix[new_col]]), axis=0)
    print(max_cols)
    return np.array(max_cols)

def compute_new_medoid(cluster, distances):
    mask = np.ones(distances.shape)
    mask[np.ix_(cluster,cluster)] = 0.
    cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
    costs = cluster_distances.sum(axis=1)
    return costs.argmin(axis=0, fill_value=10e9)


if __name__ == "__main__":
    iris = sk_data.load_iris()
    X = iris.data#[:6,:]
    y = iris.target

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    M = euclidean_distances(X)
    clusters,medoids = cluster(M, k=15)

    print(clusters)

    # Plot the training points
    plt.figure(2, figsize=(8, 6))
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,edgecolor='k')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
