from GA_Clustering import GACluster
from GA_Clustering import euclidean_distances

model = GACluster()

print("Broj klastera: {}".format(model.n_clusters_))
print("Max iteracija: {}".format(model.max_iter_))
print("Velicina populacije: {}".format(model.population_size_))
print("Centrodi klastera: {}".format(model.cluster_centers()))
print("Labele: {}".format(model.labels()))

print(euclidean_distances([0, 0], [-3, 4]))
print(euclidean_distances([12,33,-18], [-3.14, 32.917, 12.87]))