import numpy as np
import random
from numpy.linalg import norm

def euclidean_distances(X, Y):
    X_Y = np.subtract(X,Y) # X - Y
    return norm(X_Y)


#Provera da li postoji prazan klaster
#Ako postoji vraca index praznog klastera
def exitsts_empty_cluster(clusters):
    try:
        index_of_empt = list(clusters.values()).index([])
        return index_of_empt        
    except ValueError:
        return -1


#Funkcija za preracunavanje centroida
def compute_centroids(code, clusters, num_clasters):
    for i in range(num_clasters):
                print(i, clusters[i],len(clusters[i]))
                code[i] =  np.sum(clusters[i], axis=0)  / len(clusters[i])
                print('Code', code[i])
                # print(f"Klaster {i} : {nove_centroide[i]}, suma = {suma}, novi centar = { suma / len(nove_centroide[i])}")
            
class Individual:
    def __init__(self, num_clasters, points, mutation_rate, firstInit = True):
        self.num_clasters = num_clasters
        self.points = points
        self.mutation_rate = mutation_rate
        self.labels = []
        self.code = []
        if firstInit:
            self.initialize()
            
    def __lt__(self, other):
         return self.fitness() < other.fitness()
         
    def fitness(self):
        sse_distances = 0.0
        for i in range(len(self.labels)):
            # print(tacke[i], self.code[self.labels[i]], euclidean_distances(self.code[self.labels[i]],tacke[i]))
            sse_distances += euclidean_distances(self.code[self.labels[i]],self.points[i])
        # sse_distances = sum([ euclidean_distances(self.code[self.labels[i]],self.points[i]) for i in range(len(self.labels))])
        return sse_distances

    def initialize(self):
        self.code = random.sample(self.points, k = self.num_clasters)
    
    def precomputeDistances(self):
#        print(f"Pocetni centroidi: {self.code}")
        labels = []
        for i in range(len(self.points)):
            min = 0
            for j in range(1, self.num_clasters):
                if euclidean_distances(self.points[i], self.code[j]) < euclidean_distances(self.points[i], self.code[min]):
                    min = j
            labels.append(min)
        self.labels = labels

        clusters = dict() # 0 -> []
        # print ('Labels :', labels)
        for i in range(0, self.num_clasters):
            clusters[i] = [] 
                    
        for i in range(len(labels)): 
            clusters[labels[i]].append(self.points[i])

        #Azuriranje centroida
        compute_centroids(code=self.code, clusters=clusters, num_clasters=self.num_clasters)
        
        #Ovde proveravam ako je klaster przan ...
        empty_index  = exitsts_empty_cluster(clusters)
        while empty_index !=-1:
            print("\n\n------PRAZAN KLASTER--------:  ", empty_index)            
            #Uzmi random tacku i prebaci je iz njenoh klastera u prazan klaster
            p_index = random.randrange(len(self.points))
            clusters[empty_index].append(self.points[p_index])
           
            #Azuriraj labelu
            prev_cluster = labels[p_index]
            labels[p_index] = empty_index

            #Izbaci ovu tacku iz prethodnog klastera
            clusters[prev_cluster].remove(self.points[p_index])

            #Ponovo azuriraj centroide
            compute_centroids(code=self.code, clusters=clusters, num_clasters=self.num_clasters)

            #Ponovo proveri ima li praznih klastera
            empty_index = exitsts_empty_cluster(clusters)
            
#Neka bude ruletska za sad
def selection(population):
    lengthOfPopulation = len(population)
    # calcFitness = []
    # sumOfFitness = 0
    # probabilities = []

    # for i in range(lengthOfPopulation):
    #     calcFitness.append(population[i].fitness())
    #     sumOfFitness += calcFitness[i]

    # for i in range(lengthOfPopulation):
    #     probabilities.append(calcFitness[i] / sumOfFitness)

    #krace 
    calcFitness = [population[i].fitness() for i  in range(lengthOfPopulation)]
    sumOfFitness = sum(calcFitness)
    # sumOfFitness = sum(map(lambda ind: ind.fitess(), population))
    probabilities = [population[i].fitness() / sumOfFitness  for i in range(lengthOfPopulation)]        
    

    indexes = list(zip(range(lengthOfPopulation), np.cumsum(probabilities)))
    #print(indexes)
    k = 0
    
    prob = random.random()
    
    #print(prob)
    for j in range(0, lengthOfPopulation):
        if indexes[j][1] > prob or indexes[j][1] == prob :
            break
        
    #print(j)
    k = indexes[j][0]
    #print(population[k].code)
    #print(k)
    
    return k

def crossover(parent1, parent2, child1, child2):
    chromosomeLength = len(parent1.code)
    i = random.randrange(chromosomeLength)

    # for j in range(i):
    #     child1.code[j] = parent1.code[j]
    #     child2.code[j] = parent2.code[j]
   
    #krace:
    child1.code[:i], child2.code[:i] = parent1.code[:i], parent2.code[:i]

    # for j in range(i, chromosomeLength):
    #     child1.code[j] = parent2.code[j]
    #     child2.code[j] = parent1.code[j]
   
    child1.code[i:], child2.code[i:] = parent2.code[i:], parent1.code[i:]


def mutation(individual, mutation_rate):

    individualLength = len(individual)

    gene = len(individual[0])
    
    sign = -1
    if (random.random() < 0.5):
        sign = 1

    for i in range(individualLength):
        r = random.random()  
        
        if r > mutation_rate:
            continue
        delta = random.uniform(0, 1)
        for j in range(gene):
            if (individual[i][j] == 0):
                individual[i][j] = individual[i][j] + sign * 2 * delta
            else:
            	individual[i][j] = individual[i][j] + sign * 2 * delta * individual[i][j]

def genethic_algorithm(num_clasters, points, mutation_rate, pop_size, max_iter, elitism_size):
    #kreiranje inicijalne populacije
    population = [Individual(num_clasters,points, mutation_rate,firstInit=True) for _ in range(pop_size)]
    
    newPopulation = [Individual(num_clasters,points, mutation_rate,firstInit=True) for _ in range(pop_size)]
    #izracunavanje fitnesa za svaku jedinku -> neophodno je da isklasterujem tacke i kreiram nove centroide

    for individual in population:
        individual.precomputeDistances()
       

    for i in range(max_iter):
        population.sort()
        
        
        for i in range(elitism_size):
            newPopulation[i] = population[i]
            
        for i in range(elitism_size, pop_size, 2):
            
            k1 = selection(population)
            k2 = selection(population)
            
            
            #print('pp')
            #print(population[k1].code)
            #print(population[k2].code)
            
            crossover(population[k1], population[k2], newPopulation[i], newPopulation[i + 1])
            #print('new')
            #print(newPopulation[i].code)
            #print(newPopulation[i+1].code)
            
            #mutation(newPopulation[i].code, mutation_rate)
            #mutation(newPopulation[i+1].code, mutation_rate)
            
            newPopulation[i].precomputeDistances()
            newPopulation[i+1].precomputeDistances()
        
        population = newPopulation
        

    population.sort()
        
    return population[0]
