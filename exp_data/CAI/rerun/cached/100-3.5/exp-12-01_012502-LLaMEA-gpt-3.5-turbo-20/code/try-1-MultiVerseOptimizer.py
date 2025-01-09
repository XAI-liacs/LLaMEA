import numpy as np

class MultiVerseOptimizer:
    def __init__(self, budget, dim, num_universes=10, min_value=-5.0, max_value=5.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.num_universes = num_universes
        self.min_value = min_value
        self.max_value = max_value
        self.gamma = gamma

    def evolve_universes(self, universes):
        for i in range(len(universes)):
            other_universes = np.delete(universes, i, axis=0)
            rand_universe = other_universes[np.random.randint(len(other_universes))]
            j_rand = np.random.randint(self.dim)
            universes[i, j_rand] = rand_universe[j_rand]

    def __call__(self, func):
        universes = np.random.uniform(self.min_value, self.max_value, (self.num_universes, self.dim))
        fitness = np.array([func(u) for u in universes])
        
        for _ in range(self.budget):
            sorted_indices = np.argsort(fitness)
            alpha = 1.0 - (_ + 1) * ((1.0) / self.budget) ** self.gamma
            for i in range(self.num_universes):
                new_universe = universes[i] + alpha * (universes[sorted_indices[0]] - np.abs(universes[i]))
                new_universe = np.clip(new_universe, self.min_value, self.max_value)
                if func(new_universe) < fitness[i]:
                    universes[i] = new_universe
                    fitness[i] = func(new_universe)
                self.evolve_universes(universes)
        
        best_universe = universes[np.argmin(fitness)]
        return func(best_universe)