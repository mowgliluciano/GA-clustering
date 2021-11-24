import numpy as np
from scipy.spatial import distance
import random
import numpy.linalg as LA
import matplotlib.pyplot as plt


def euclidean_distances(X, Y):
    X_Y = np.subtract(X,Y) # X - Y
    
    return LA.norm(X_Y)

def manhattan_distances(X, Y):
    return sum(abs(val1-val2) for val1, val2 in zip(X, Y))


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
    def __init__(self, num_clasters, points, mutation_rate, distance, firstInit = True):
        self.num_clasters = num_clasters
        self.points = points
        self.mutation_rate = mutation_rate
        self.distance = distance
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
            if self.distance == "euclidean":
                sse_distances += euclidean_distances(self.code[self.labels[i]],self.points[i])
            elif self.distance == "manhattan":
                sse_distances += manhattan_distances(self.code[self.labels[i]],self.points[i])
            else:
                raise ValueError('Distance must be euclidean or manhattan.')
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
                if self.distance == "euclidean":
                    if euclidean_distances(self.points[i], self.code[j]) < euclidean_distances(self.points[i], self.code[min]):
                        min = j
                elif self.distance == "manhattan":
                    if manhattan_distances(self.points[i], self.code[j]) < manhattan_distances(self.points[i], self.code[min]):
                        min = j
                else:
                    raise ValueError('Distance must be euclidean or manhattan.')
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

def selection(population, category='roulette', TOURNAMENT_SIZE=2):
    
    if category == 'roulette':
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
        k = 0
    
        prob = random.random()
    
        #print(prob)
        for j in range(0, lengthOfPopulation):
            if indexes[j][1] > prob or indexes[j][1] == prob :
                break
        
        #print(j)
        k = indexes[j][0]
        
    elif category == 'tournament':
        
        bestFitness = float('inf')
        k = -1
    
        for i in range(TOURNAMENT_SIZE):
            index = random.randrange(len(population))
            if population[index].fitness() < bestFitness:
                bestFitness = population[index].fitness()
                k = index
    else:
        raise ValueError('Category must be tournament or roulette')
        
        
        
    
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
        
    r = random.random()  
        
    if r > mutation_rate:
        return

    randomCentroid = random.randrange(individualLength)
        
    delta = random.uniform(0, 1)
    for j in range(gene):
        if (individual[randomCentroid][j] == 0):
            individual[randomCentroid][j] = individual[randomCentroid][j] + sign * 2 * delta
        else:
            individual[randomCentroid][j] = individual[randomCentroid][j] + sign * 2 * delta * individual[randomCentroid][j]

def genethic_algorithm(num_clasters, points, mutation_rate, pop_size, category, distance,  max_iter, elitism_size):
    #kreiranje inicijalne populacije
    population = [Individual(num_clasters,points, mutation_rate,distance, firstInit=True) for _ in range(pop_size)]
    
    newPopulation = [Individual(num_clasters,points, mutation_rate, distance, firstInit=True) for _ in range(pop_size)]
    #izracunavanje fitnesa za svaku jedinku -> neophodno je da isklasterujem tacke i kreiram nove centroide

    for individual in population:
        individual.precomputeDistances()
        
    plt_ind = 1
    fig = plt.figure(figsize=(20, 20))
       
    colors = np.array(['red', 'blue', 'gold',  'green', 'plum', 'orange', 'magenta'])
    
    if max_iter > 20:
        step =  20
    else:
        step =  5
        
    n = max_iter / step + 1
        
    
    for k in range(max_iter):
        population.sort()
        if (k+1) % 10 == 0 or k+1 == max_iter:
            
            fig.add_subplot(n, 2, plt_ind)
            
            labels  = population[0].labels
            centres = population[0].code
            
            
            plt.scatter(x=list(map(lambda point: point[0],population[0].points)), y=list(map(lambda point: point[1],population[0].points)), c=colors[labels])
            
            plt.scatter(x=list(map(lambda centre: centre[0], centres)), y=list(map(lambda centre: centre[1], centres)), c=['black'], marker='s')

            #postavljanje naslova za svaku celiju
            plt.title("iteration: %d     inertia: %d   %s  %s"  % (k, population[0].fitness(), distance, category), fontsize=10)

            #prelazak u narednu celiju 
            plt_ind+=1
            

          
        
        for i in range(elitism_size):
            newPopulation[i] = population[i]
            
        for i in range(elitism_size, pop_size, 2):
            
            k1 = selection(population, category)
            k2 = selection(population, category)
            
            
            crossover(population[k1], population[k2], newPopulation[i], newPopulation[i + 1])

            
            mutation(newPopulation[i].code, mutation_rate)
            mutation(newPopulation[i+1].code, mutation_rate)
            
            newPopulation[i].precomputeDistances()
            newPopulation[i+1].precomputeDistances()
        
        population = newPopulation
        
        

    population.sort()
    
    plt.tight_layout()
    plt.show()
        
    return population[0]
