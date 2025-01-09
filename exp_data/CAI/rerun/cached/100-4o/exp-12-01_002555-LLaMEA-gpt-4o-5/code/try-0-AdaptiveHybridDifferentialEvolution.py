import numpy as np

class AdaptiveHybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.eval_count = 0

    def evaluate(self, func, candidate):
        fitness = func(candidate)
        self.eval_count += 1
        return fitness

    def __call__(self, func):
        self.fitness = np.array([self.evaluate(func, ind) for ind in self.population])
        best_index = np.argmin(self.fitness)
        best_individual = self.population[best_index]

        while self.eval_count < self.budget:
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), -5.0, 5.0)
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])
                trial_fitness = self.evaluate(func, trial)

                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    self.population[i] = trial
                    if trial_fitness < self.fitness[best_index]:
                        best_index = i
                        best_individual = trial
            
            # Local search on best individual
            if self.eval_count + self.dim <= self.budget:
                local_neighbors = best_individual + np.random.uniform(-0.1, 0.1, (self.dim, self.dim))
                local_neighbors = np.clip(local_neighbors, -5.0, 5.0)
                for neighbor in local_neighbors:
                    local_fitness = self.evaluate(func, neighbor)
                    if local_fitness < self.fitness[best_index]:
                        best_index = np.argmin(local_fitness)
                        best_individual = neighbor

        return best_individual