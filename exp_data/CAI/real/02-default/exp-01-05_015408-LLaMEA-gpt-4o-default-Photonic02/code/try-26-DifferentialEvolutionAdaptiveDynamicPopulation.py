import numpy as np

class DifferentialEvolutionAdaptiveDynamicPopulation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population = None
        self.fitness = None
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.initial_population_size, self.dim))
        self.fitness = np.array([func(ind) for ind in self.population])
        evaluations = self.initial_population_size
        best_index = np.argmin(self.fitness)
        best_position = self.population[best_index]

        while evaluations < self.budget:
            new_population = np.copy(self.population)
            for i in range(len(self.population)):
                a, b, c = self.select_three_unique(i, len(self.population))
                mutant_vector = self.mutate(self.population[a], self.population[b], self.population[c], lb, ub)
                trial_vector = self.crossover(self.population[i], mutant_vector)
                
                new_fitness = func(trial_vector)
                evaluations += 1

                if new_fitness < self.fitness[i]:
                    new_population[i] = trial_vector
                    self.fitness[i] = new_fitness

                if new_fitness < self.fitness[best_index]:
                    best_index = i
                    best_position = trial_vector

                if evaluations >= self.budget:
                    break

            self.population = new_population
            self.dynamic_population_adjustment(evaluations)

        return best_position, self.fitness[best_index]

    def select_three_unique(self, idx, population_size):
        indices = [i for i in range(population_size) if i != idx]
        return np.random.choice(indices, 3, replace=False)

    def mutate(self, a, b, c, lb, ub):
        mutant = a + self.F * (b - c)
        return np.clip(mutant, lb, ub)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        return np.where(crossover_mask, mutant, target)

    def dynamic_population_adjustment(self, evaluations):
        if evaluations < self.budget * 0.5:
            self.population_size = self.initial_population_size
        elif evaluations < self.budget * 0.75:
            self.population_size = self.initial_population_size // 2
        else:
            self.population_size = self.initial_population_size // 4

        self.population = self.population[:self.population_size]
        self.fitness = self.fitness[:self.population_size]