from GA_Clustering import GACluster
import ga

# jedinka = ga.Individual(brojKlastera, tacke, 0.05)
# genethic_algorithm(brojKlastera,tacke, 0.05, 10, 100, 10)

brojKlastera = 3
tacke =  [[6.5, 0], [9, 0], [10, 0], [15, 0], [16, 0], [18.5, 0]] 
model  = GACluster(brojKlastera, max_iter=10, population_size=19, mutation_rate=0.05)
print("Tacke {}".format(tacke))
model.fit(tacke)
print(f"Labele: {model.labels()}")
print(f"Centroidi: {model.cluster_centers()}")
print("SSE: {}".format(model.sse()))
