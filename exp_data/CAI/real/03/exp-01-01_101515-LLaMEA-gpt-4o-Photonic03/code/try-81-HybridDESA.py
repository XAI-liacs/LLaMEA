import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_prob = 0.9
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += self.population_size
        
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant_vector = self.adaptive_mutate(a, b, c, bounds)  # Changed mutation method
                trial_vector = self.adaptive_crossover(population[i], mutant_vector)  # Changed crossover method
                
                trial_fitness = func(trial_vector)
                self.evaluations += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness
                    
                if self.evaluations >= self.budget:
                    break
        
        best_index = np.argmin(fitness)
        return population[best_index]

    def adaptive_mutate(self, a, b, c, bounds):
        adapt_factor = np.random.uniform(0.3, np.random.uniform(0.8, 0.95))  # Adjusted adaptation factor range with randomness
        mutant = a + adapt_factor * (b - c)
        return np.clip(mutant, bounds[:, 0], bounds[:, 1])

    def adaptive_crossover(self, target, mutant):
        adapt_prob = np.random.uniform(0.8, 1.0)  # Adaptive crossover probability
        cross_points = np.random.rand(self.dim) < adapt_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial