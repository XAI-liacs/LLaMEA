import numpy as np

class AdaptiveClusterDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(10 * dim, budget // 2)
        self.f = 0.5
        self.cr = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        
        evaluations = self.pop_size
        
        while evaluations < self.budget:
            # Cluster and select leaders
            cluster_indices = np.random.randint(0, 2, self.pop_size)
            leaders = [population[cluster_indices == i][np.argmin(fitness[cluster_indices == i])] for i in range(2)]
            
            # Generate trial vectors
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                leader = leaders[cluster_indices[i]]
                mutant = np.clip(a + self.f * (b - c + leader - population[i]), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.cr
                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluate trial vector
                trial_fitness = func(trial)
                evaluations += 1

                # Select between trial and original
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                
                if evaluations >= self.budget:
                    break

        best_index = np.argmin(fitness)
        return population[best_index]

# Example of usage:
# optimizer = AdaptiveClusterDE(budget=1000, dim=5)
# result = optimizer(some_function)