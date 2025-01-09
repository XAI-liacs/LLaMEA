import numpy as np

class QFireworksAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        n_explosions = 5
        n_sparks = 10
        pop_size = n_explosions * n_sparks
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        for _ in range(self.budget):
            best_idx = np.argmin(fitness)
            best = pop[best_idx]
            for i in range(n_explosions):
                center = pop[i * n_sparks]
                for j in range(1, n_sparks):
                    pop[i * n_sparks + j] = center + np.random.normal(0, 1, self.dim) * np.abs(best - center)
                    fitness[i * n_sparks + j] = func(pop[i * n_sparks + j])
        return best