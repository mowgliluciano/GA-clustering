import numpy as np
import ga
class GACluster:
    # ctor 
    def __init__(self, n_clusters=8, max_iter = 500, population_size = 100, category='tournament', distance = 'euclidean', mutation_rate = 0.05, elitism_size = 1):
        
        self.n_clusters_ = n_clusters
        self.max_iter_ = max_iter
        self.population_size_ = population_size
        self.category_ = category
        self.distance_ = distance
        self.mutation_rate_ = mutation_rate
        self.elitism_size_ = elitism_size
        self.cluster_centers_ = np.ndarray(0)
        self.labels_ = np.array([])
        self.sse_ = float('inf')
        
    
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
        return self.sse_
    
    #X : ndarray 
    #Compute k-means clustering
    def fit(self, X):
        individual = ga.genethic_algorithm(self.n_clusters_, X, self.mutation_rate_, 
                                           self.population_size_, self.category_, self.distance_, self.max_iter_, self.elitism_size_)
        self.cluster_centers_ = individual.code
        self.labels_ = individual.labels
        self.sse_ = individual.fitness()
