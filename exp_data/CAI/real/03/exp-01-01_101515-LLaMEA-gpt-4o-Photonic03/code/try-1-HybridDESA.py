import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_population_size = 8 * dim  # Reduced initial population size
        self.population_size = self.base_population_size
        self.mutation_factor = 0.5
        self.crossover_prob = 0.9
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += self.population_size
        
        while self.evaluations < self.budget:
            if self.evaluations % (self.budget // 5) == 0:
                self.mutation_factor = 0.3 + 0.7 * np.random.rand()  # Dynamic scaling factor
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant_vector = self.mutate(a, b, c, bounds)
                trial_vector = self.crossover(population[i], mutant_vector)
                
                trial_fitness = func(trial_vector)
                self.evaluations += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                if self.evaluations % (self.budget // 3) == 0:
                    self.population_size = min(self.base_population_size * 2, self.population_size + self.base_population_size // 2)
                    population = np.vstack([population, np.random.rand(self.base_population_size // 2, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]])
                    fitness = np.hstack([fitness, np.array([func(ind) for ind in population[-(self.base_population_size // 2):]])])
                    self.evaluations += self.base_population_size // 2

                if self.evaluations >= self.budget:
                    break
        
        best_index = np.argmin(fitness)
        return population[best_index]

    def mutate(self, a, b, c, bounds):
        mutant = a + self.mutation_factor * (b - c)
        return np.clip(mutant, bounds[:, 0], bounds[:, 1])

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial