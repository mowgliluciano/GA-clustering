import ga
class GACluster:
    def __init__(self, n_clusters=8, max_iter = 500, population_size = 100, category='roulette',
                 tournament_size=2, distance = 'ss_dist', mutation_rate = 0.05, elitism_size = 1):
        self.n_clusters_ = n_clusters
        self.max_iter_ = max_iter
        self.population_size_ = population_size
        self.category_ = category
        self.tournament_size = tournament_size

        if distance not in ['ss_dist', 'manhattan']:
            raise  ValueError('Disttance must be ss_dist or manhattan')
        self.distance_ = distance
        
        self.mutation_rate_ = mutation_rate
        self.elitism_size_ = elitism_size
        self.cluster_centers_ = []
        self.labels_ = []
        self.inertia_ = float('inf')
        
    
    # cluster_centers_ :ndarray of shape (n_clusters, n_features)
    def cluster_centers(self):
        return self.cluster_centers_

    #labels_ : ndarray of shape (n_samples,)
    #Labels of each point
    def labels(self):
        return self.labels_
    
    # inertia_ : float
    # Sum of squared distances of samples to their closest cluster center.
    def inertia(self):
        return self.inertia_
    
    #X : list of tuples 
    #Compute k-means clustering
    def fit(self, X):
        individual = ga.genethic_algorithm(self.n_clusters_, X, self.mutation_rate_, 
                                           self.population_size_, self.category_, self.tournament_size, 
                                           self.distance_, self.max_iter_, self.elitism_size_)
        
        self.cluster_centers_ = individual.code
        self.labels_ = individual.labels
        self.inertia_ = individual.fitness_
