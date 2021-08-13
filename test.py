from GA_Clustering import GACluster
import ga

# jedinka = ga.Individual(brojKlastera, tacke, 0.05)
# genethic_algorithm(brojKlastera,tacke, 0.05, 10, 100, 10)

brojKlastera = 3
tacke =  [[1,1], [12.2,2],[10.34,3],[0,0],[6,6]] 
model  = GACluster(brojKlastera, max_iter=100, population_size=20, mutation_rate=0.05)
model.fit(tacke)
print("Tacke {}".format(tacke))
print(f"Labele: {model.labels()}")
print(f"Centroidi: {model.cluster_centers()}")
print("SSE: {}".format(model.sse()))
