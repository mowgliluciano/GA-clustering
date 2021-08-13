import numpy as np
import random
from numpy.linalg import norm

def euclidean_distances(X, Y):
    X_Y = np.subtract(X,Y) # X - Y
    return norm(X_Y)

class Individual:
    def __init__(self, num_clasters, points, mutation_rate, firstInit = True):
        self.num_clasters = num_clasters
        self.points = points
        self.mutation_rate = mutation_rate
        self.labels = []
        self.code = []
        if firstInit:
            self.initialize()

    def fitness(self):
        sse_distances = 0.0
        for i in range(len(self.labels)):
            # print(tacke[i], self.code[self.labels[i]], euclidean_distances(self.code[self.labels[i]],tacke[i]))
            sse_distances += euclidean_distances(self.code[self.labels[i]],self.points[i])
        return sse_distances

    def initialize(self):
        self.code = random.sample(self.points, k = self.num_clasters)
    
    def precomputeDistances(self):
        # print(f"Pocetni centroidi: {self.code}")
        labels = []
        for i in range(len(self.points)):
            min = 0
            for j in range(1, self.num_clasters):
                if euclidean_distances(self.points[i], self.code[j]) < euclidean_distances(self.points[i], self.code[min]):
                    min = j
            labels.append(min)
            self.labels = labels
       
        clusters = dict()
        for i in range(0, self.num_clasters):
            clusters[i] = []

        for i in range(len(labels)): 
            clusters[labels[i]].append(self.points[i])

        for i in range(0,self.num_clasters):
            self.code[i] =  np.sum(clusters[i], axis=0)  / len(clusters[i])
            # print(f"Klaster {i} : {nove_centroide[i]}, suma = {suma}, novi centar = { suma / len(nove_centroide[i])}")


def genethic_algorithm(num_clasters, points, mutation_rate, pop_size, max_iter, elitism_size):
    #kreiranje inicijalne populacije
    population = [Individual(num_clasters,points, mutation_rate,firstInit=True) for _ in range(pop_size)]
    #izracunavanje fitnesa za svaku jedinku -> neophodno je da isklasterujem tacke i kreiram nove centroide
    for individual in population:
        individual.precomputeDistances()
        # print(individual.fitness())

    # for i in range(max_iter):
        #selekcija...
        #ukrstanje...
        #mutacija...
        #elitizam...
        #smena generacija...
    return population[0]
