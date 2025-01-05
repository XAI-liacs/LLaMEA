import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.adaptive_factor = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_individual = population[best_index]

        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices[0]], population[indices[1]], population[indices[2]]
                mutant = x1 + self.F * (x2 - x3)
                mutant = np.clip(mutant, lb, ub)

                trial = np.copy(population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]

                new_fitness = func(trial)
                evaluations += 1

                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    population[i] = trial

                    if new_fitness < fitness[best_index]:
                        best_index = i
                        best_individual = trial

                if evaluations >= self.budget:
                    break

            # Adaptive strategy adjustments
            self.adaptive_strategy(fitness, best_index)

        return best_individual, fitness[best_index]

    def adaptive_strategy(self, fitness, best_index):
        # Adjust F and CR based on progress
        progress = (np.max(fitness) - fitness[best_index]) / (np.max(fitness) + 1e-9)
        self.F = np.clip(self.F + self.adaptive_factor * (0.5 - progress), 0.1, 1.0)
        self.CR = np.clip(self.CR + self.adaptive_factor * (0.5 - progress), 0.1, 1.0)