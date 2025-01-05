import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Rule of thumb
        self.mutation_factor = 0.5
        self.crossover_prob = 0.9
        self.evaluations = 0

    def __call__(self, func):
        # Initialize population within bounds
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += self.population_size
        
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Dynamic adaptation of the mutation factor based on evaluations
                self.mutation_factor = 0.8 - 0.3 * (self.evaluations / self.budget)
                # Select three random indices different from i
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                # Perform mutation and crossover
                mutant_vector = self.mutate(a, b, c, bounds)
                trial_vector = self.crossover(population[i], mutant_vector)
                
                # Evaluate the trial vector
                trial_fitness = func(trial_vector)
                self.evaluations += 1
                
                # Selection process
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness
                else:
                    # Local search around the best known solution
                    local_search_vector = population[i] + np.random.uniform(-0.1, 0.1, self.dim)
                    local_search_vector = np.clip(local_search_vector, bounds[:, 0], bounds[:, 1])
                    local_fitness = func(local_search_vector)
                    self.evaluations += 1
                    if local_fitness < fitness[i]:
                        population[i] = local_search_vector
                        fitness[i] = local_fitness
                
                if self.evaluations >= self.budget:
                    break
        
        # Return the best solution found
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