import numpy as np
from numpy.linalg import norm

def euclidean_distances(X, Y):
    X_Y = np.subtract(X,Y) # X - Y
    return norm(X_Y)


class GACluster:
    # ctor 
    def __init__(self, n_clusters=8, max_iter = 500, population_size = 100):
        self.n_clusters_ = n_clusters
        self.max_iter_ = max_iter
        self.cluster_centers_ = np.ndarray(0)
        self.labels_ = np.ndarray(0)
        self.population_size_ = population_size
    
    # cluster_centers_ :ndarray of shape (n_clusters, n_features)
    def cluster_centers(self):
        return self.cluster_centers_

    #labels_ : ndarray of shape (n_samples,)
    #Labels of each point
    def labels(self):
        return self.labels_
    
    # inertia_ : float
    # Sum of squared distances of samples to their closest cluster center.
    def sse(self):
        pass
    
    #X : ndarray 
    #Compute k-means clustering
    def fit(X):
        pass
