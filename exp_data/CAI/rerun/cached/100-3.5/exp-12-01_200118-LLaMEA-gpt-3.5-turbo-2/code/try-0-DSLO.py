import numpy as np

class DSLO:
    def __init__(self, budget, dim, n_pop=50, n_neigh=5, alpha=0.1):
        self.budget = budget
        self.dim = dim
        self.n_pop = n_pop
        self.n_neigh = n_neigh
        self.alpha = alpha

    def _initialize_population(self):
        return np.random.uniform(-5.0, 5.0, size=(self.n_pop, self.dim))

    def _evaluate_fitness(self, population, func):
        return np.apply_along_axis(func, 1, population)

    def _get_neighbors(self, idx, population):
        distances = np.linalg.norm(population - population[idx], axis=1)
        sorted_neighbors = np.argsort(distances)
        return sorted_neighbors[1:self.n_neigh+1]

    def __call__(self, func):
        population = self._initialize_population()
        fitness = self._evaluate_fitness(population, func)
        
        for _ in range(self.budget - self.n_pop):
            for i in range(self.n_pop):
                neighbors = self._get_neighbors(i, population)
                best_neighbor = neighbors[np.argmin(fitness[neighbors])]
                social_mean = np.mean(population[neighbors], axis=0)
                population[i] = (1 - self.alpha) * population[i] + self.alpha * social_mean
                fitness[i] = func(population[i])
                
        best_idx = np.argmin(fitness)
        return population[best_idx]