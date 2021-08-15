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
            
            
def selection(population):
    lengthOfPopulation = len(population)

    calcFitness = np.zeros(lengthOfPopulation)
    sumOfFitness = 0
   # probabilities = np.zeros(lengthOfPopulation)

    for i in range(lengthOfPopulation):
        calcFitness[i] = population[i].fitness()
        sumOfFitness += calcFitness[i]

    #for i in range(lengthOfPopulation):
    #    probabilities[i] = calcFitness[i] / sumOfFitness
    #    print(probabilities[i], i)

    indexes = sorted(list(zip(range(lengthOfPopulation), calcFitness)), key=lambda x: x[1])
    #print(indexes)
    k = 0
    prob = random.random()
    #print(prob)
    for j in range(0, lengthOfPopulation):

        if indexes[j][1] > prob:
            break

        #print(j)
        k = indexes[j][0]
        #print(k)

    return k

def crossover(parent1, parent2):
    chromosomeLength = len(parent1.code)

    child1 = parent1
    child2 = parent2

    i = random.randrange(chromosomeLength)

    for j in range(i):
        child1.code[j] = parent1.code[j]
        child2.code[j] = parent2.code[j]

    for j in range(i, chromosomeLength):
        child1.code[j] = parent2.code[j]
        child2.code[j] = parent1.code[j]

    return child1, child2


def mutation(individual, number):
    chromosomeLength = len(individual.code)
    gene = len(individual.code[0])
    sign = -1
    if (random.random() < 0.5):
        sign = 1

    for i in range(chromosomeLength):
        r = random.random()
        print(r)
        if r > number:
            continue
        delta = random.uniform(0, 1)
        for j in range(gene):
            if (individual.code[i][j] == 0):
                individual.code[i][j] = sign * 2 * delta
        individual.code[i][j] += sign * 2 * delta * individual.code[i][j]

    return individual



def genethic_algorithm(num_clasters, points, mutation_rate, pop_size, max_iter, elitism_size):
    #kreiranje inicijalne populacije
    population = [Individual(num_clasters,points, mutation_rate,firstInit=True) for _ in range(pop_size)]
    newPopulation = [Individual(num_clasters, points, mutation_rate, firstInit=True) for _ in range(pop_size)]
    #izracunavanje fitnesa za svaku jedinku -> neophodno je da isklasterujem tacke i kreiram nove centroide
    for individual in population:
        individual.precomputeDistances()
        # print(individual.fitness())

    for i in range(pop_size):
        newPopulation[i] = population[i]

    for i in range(max_iter):
        for j in range(pop_size-1):
            k1 = selection(population)
            k2 = selection(population)
            newPopulation[j], newPopulation[j+1] = crossover(population[k1], population[k2])
            #mutation(newPopulation[j], mutation_rate)
        # mutacija...
        # elitizam...
        # smena generacija...

        population = newPopulation
        #elitizam...
        #smena generacija...
    return population[0]
