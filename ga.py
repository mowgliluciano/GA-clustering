import numpy as np
import random
import matplotlib.pyplot as plt


def ss_dist(X, Y):
    return np.sum(np.subtract(X,Y)**2)

def manhattan_distances(X, Y):
    return sum(abs(val1-val2) for val1, val2 in zip(X, Y))

# Funkcija koja vraca indeks praznog klastera
def exitsts_empty_cluster(clusters):
    try:
        index_of_empt = clusters.index([])
        return index_of_empt        
    except ValueError:
        return -1
        
# Funkcija za preracunavanje centroida
def compute_centroids(code, clusters, num_clasters, points):
    for i in range(num_clasters):
        code[i] =  np.sum(points[clusters[i]], axis=0)  / len(clusters[i])
    
    
# Funkcija koja iscrtava klasterovanje najbolje jedinke      
def plot_best_individual(currBestIndividual, k, category, distance):
    colors = np.array(['red', 'blue', 'gold',  'green', 'plum', 'orange', 'magenta'])
    #Iscrtavanje tacaka
    plt.scatter(x=currBestIndividual.points[ : , 0],
                y=currBestIndividual.points[ : , 1],
                c=colors[currBestIndividual.labels])
    
    # Iscrtavanje centara
    plt.scatter(x=currBestIndividual.code[:, 0],
                y=currBestIndividual.code[:, 1],
                c=['black'], marker='x')
    
    # Postavljanje naslova za svaku celiju
    plt.title(label="iteration: {}     inertia: {}   {}  {}".format(k, int(currBestIndividual.fitness_),
                distance, category), fontsize=10)


def  rouletteSelection(population):
    calcFitness = [population[i].fitness_ for i  in range(len(population))]
    fitnessSum = sum(calcFitness)
    probabilities = [population[i].fitness_ / fitnessSum  for i in range(len(population))]        
    cumSumProb = np.cumsum(probabilities)
    
    indexes = list(zip(range(len(population)), cumSumProb))
    k = 0

    prob = random.random()
    for j in range(len(population)):
        if indexes[j][1] > prob or indexes[j][1] == prob :
            break
    
    k = indexes[j][0]
    return k


def tournamentSelection(population, TOURNAMENT_SIZE):
    bestFitness = float('inf')
    k = -1

    for _ in range(TOURNAMENT_SIZE):
        index = random.randrange(len(population))
        if population[index].fitness_ < bestFitness:
            bestFitness = population[index].fitness_
            k = index
    
    return k

class Individual:
    def __init__(self, num_clasters, points, mutation_rate, distance):
        self.num_clasters = num_clasters
        self.points = np.array(points)
        self.mutation_rate = mutation_rate
        self.distance = distance
        self.labels = []
        self.code = []
        self.fitness_ = 0
        self.initialize()
            
    def __lt__(self, other):
         return self.fitness_ < other.fitness_
         
    def fitness(self):
        distances = 0.0
        for i in range(len(self.labels)):
            if self.distance == "ss_dist":
                distances += ss_dist(self.code[self.labels[i]],self.points[i])
            else:
                distances += manhattan_distances(self.code[self.labels[i]],self.points[i])
        self.fitness_ = distances

    def initialize(self):
        self.code = np.array(random.sample(list(self.points), k = self.num_clasters))
        self.precomputeDistances()
           
    def precomputeDistances(self):
        labels = []
        clusters = [ [] for _ in range(self.num_clasters) ]
        for i in range(len(self.points)):
            min = 0
            for j in range(1, self.num_clasters):
                if self.distance == "ss_dist":
                    if ss_dist(self.points[i], self.code[j]) < ss_dist(self.points[i], self.code[min]):
                        min = j
                else:
                    if manhattan_distances(self.points[i], self.code[j]) < manhattan_distances(self.points[i], self.code[min]):
                        min = j
            labels.append(min)
            clusters[min].append(i)

        self.labels = labels
        

        # Ovde proveravam ako je klaster prazan ...
        empty_index  = exitsts_empty_cluster(clusters)
        while empty_index !=-1:
            r_index = random.randrange(len(self.points))
            clusters[empty_index].append(r_index)

            # Izbaci ovu tacku iz prethodnog klastera
            prev_cluster = self.labels[r_index]
            clusters[prev_cluster].remove(r_index)

            # Azuriraj labelu
            self.labels[r_index] = empty_index
            
            # Ponovo proveri ima li praznih klastera
            empty_index = exitsts_empty_cluster(clusters)
        
        # Preracunavanje  centroida
        compute_centroids(code=self.code, clusters=clusters, num_clasters=self.num_clasters, points=self.points)
       
        #Azuriraj fitnes
        self.fitness()
        
def selection(population, category, TOURNAMENT_SIZE):
    if category == 'roulette':
        k = rouletteSelection(population)
    else:
        k = tournamentSelection(population, TOURNAMENT_SIZE)
    
    return k

def crossover(parent1, parent2, child1, child2):
    i = random.randrange(len(parent1.code))    
    child1.code[:i], child2.code[:i] = parent1.code[:i], parent2.code[:i]
    child1.code[i:], child2.code[i:] = parent2.code[i:], parent1.code[i:]


def mutation(individual, mutation_rate):
    r = random.random()  
    if r > mutation_rate:
        return
    
    individualLength = len(individual)
    gene = len(individual[0])
    
    sign = -1
    if (random.random() < 0.5):
        sign = 1
        
    randomCentroid = random.randrange(individualLength)
    delta = random.uniform(0, 1)
    for j in range(gene):
        if (individual[randomCentroid][j] == 0):
            individual[randomCentroid][j] = individual[randomCentroid][j] + sign * 2 * delta
        else:
            individual[randomCentroid][j] = individual[randomCentroid][j] + sign * 2 * delta * individual[randomCentroid][j]


# *****************************************************************************************************************

def genethic_algorithm(num_clasters, points, mutation_rate, pop_size,
                       category, tournament_size, distance,  max_iter, elitism_size):
    population = [Individual(num_clasters,points, mutation_rate,distance) for _ in range(pop_size)]
    newPopulation = [Individual(num_clasters,points, mutation_rate,distance) for _ in range(pop_size)]

    plt_ind = 1
    n = max_iter // 10
     
    fig = plt.figure(figsize=(10, n/2*5))
    if n < 4:
        fig = plt.figure(figsize=(10,5))
    
    for k in range(max_iter):
        population.sort()
        
        if k % 10 == 0:
            fig.add_subplot(int(n/2+1), 2, plt_ind)
            plot_best_individual(population[0], k, category, distance)
            #prelazak u narednu celiju 
            plt_ind += 1
            
        # Elitizam
        newPopulation[:elitism_size] = population[:elitism_size]
        
        for i in range(elitism_size, pop_size, 2):
            k1 = selection(population, category, tournament_size)
            k2 = selection(population, category, tournament_size)
            
            crossover(population[k1], population[k2], newPopulation[i], newPopulation[i + 1])

            mutation(newPopulation[i].code, mutation_rate)
            mutation(newPopulation[i+1].code, mutation_rate)
            
            # Azuriraju se centroide    
            newPopulation[i].precomputeDistances()
            newPopulation[i+1].precomputeDistances()
        
        population = newPopulation
        
        
    bestIndividual = min(population)
    #Iscrtavanje finalnog rezultata
    fig.add_subplot(int(n/2+1), 2, plt_ind)
    plot_best_individual(bestIndividual, max_iter - 1, category, distance)    
    plt.tight_layout()
    plt.show()
        
    return bestIndividual

# *****************************************************************************************************************