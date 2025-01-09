import numpy as np

class HybridDELS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.evaluations = 0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Differential Evolution Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + self.F * (x1 - x2), self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                self.evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Check budget
                if self.evaluations >= self.budget:
                    break

            # Local Search on the best individual
            if self.evaluations < self.budget:
                best_idx = np.argmin(fitness)
                best_individual = population[best_idx]
                neighborhood_size = 0.1
                for _ in range(5):  # 5 local search steps
                    if self.evaluations >= self.budget:
                        break
                    local_move = np.random.uniform(-neighborhood_size, neighborhood_size, self.dim)
                    neighbor = np.clip(best_individual + local_move, self.lower_bound, self.upper_bound)
                    neighbor_fitness = func(neighbor)
                    self.evaluations += 1
                    if neighbor_fitness < fitness[best_idx]:
                        population[best_idx] = neighbor
                        fitness[best_idx] = neighbor_fitness

        # Return the best found solution
        best_idx = np.argmin(fitness)
        return population[best_idx]