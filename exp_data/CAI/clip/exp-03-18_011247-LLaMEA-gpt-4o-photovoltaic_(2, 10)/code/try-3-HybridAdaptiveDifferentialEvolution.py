import numpy as np

class HybridAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.population = np.random.rand(self.population_size, dim)
        self.performance_memory = []
        self.current_evaluations = 0
        self.learning_rate = 0.1  # New attribute for dynamic learning

    def adaptive_mutation_factor(self):
        if len(self.performance_memory) < 2:
            return self.mutation_factor
        improvement = self.performance_memory[-1] - self.performance_memory[-2]
        return max(0.3, min(0.9, self.mutation_factor + self.learning_rate * np.sign(improvement)))

    def select_parents(self, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        return np.random.choice(indices, 3, replace=False)

    def mutate_and_crossover(self, target_idx, func):
        a, b, c = self.select_parents(target_idx)
        mutant_vector = self.population[a] + self.adaptive_mutation_factor() * (self.population[b] - self.population[c])
        mutant_vector = np.clip(mutant_vector, func.bounds.lb, func.bounds.ub)
        trial_vector = np.copy(self.population[target_idx])
        crossover_points = np.random.rand(self.dim) < self.crossover_rate
        trial_vector[crossover_points] = mutant_vector[crossover_points]
        return trial_vector

    def optimize(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        self.current_evaluations += self.population_size
        for _ in range(self.budget - self.current_evaluations):
            for i in range(self.population_size):
                trial_vector = self.mutate_and_crossover(i, func)
                trial_fitness = func(trial_vector)
                self.current_evaluations += 1
                if trial_fitness < fitness[i]:
                    self.population[i] = trial_vector
                    fitness[i] = trial_fitness
            self.performance_memory.append(np.min(fitness))
        return self.population[np.argmin(fitness)]

    def __call__(self, func):
        return self.optimize(func)