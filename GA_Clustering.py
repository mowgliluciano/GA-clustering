import numpy as np

class GACluster:
    # ctor 
    def __init__(self, n_clusters=8, max_iter = 500, population_size = 100):
        self.n_clusters_ = n_clusters
        self.max_iter_ = max_iter
        self.cluster_centers_ = [];
        self.labels_ = []
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