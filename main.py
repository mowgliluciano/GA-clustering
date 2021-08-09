from GA_Clustering import GACluster
model = GACluster()

print(model.n_clusters_)
print(model.max_iter_)
print(model.population_size_)
print(model.cluster_centers())
print(model.labels())

