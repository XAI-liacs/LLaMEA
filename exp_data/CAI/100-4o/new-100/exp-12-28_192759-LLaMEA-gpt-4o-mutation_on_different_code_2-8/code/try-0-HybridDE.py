import numpy as np

class HybridDE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * self.dim
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability

    def _local_search(self, x, func):
        step_size = 0.1
        best_x = np.copy(x)
        best_f = func(x)
        for d in range(self.dim):
            for direction in [-1, 1]:
                neighbor = np.copy(x)
                neighbor[d] += direction * step_size
                neighbor_f = func(neighbor)
                if neighbor_f < best_f:
                    best_f = neighbor_f
                    best_x = neighbor
        return best_x, best_f

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                trial = np.copy(population[i])
                
                # Binomial crossover
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == self.dim - 1:
                        trial[j] = mutant[j]
                
                # Evaluate trial vector
                trial_f = func(trial)
                
                # Replacement
                if trial_f < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_f
                    
                # Local search around the improved solution
                if trial_f < self.f_opt:
                    self.f_opt, self.x_opt = self._local_search(trial, func)
                    fitness[i] = self.f_opt

        # Determine the best solution found
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        
        return self.f_opt, self.x_opt