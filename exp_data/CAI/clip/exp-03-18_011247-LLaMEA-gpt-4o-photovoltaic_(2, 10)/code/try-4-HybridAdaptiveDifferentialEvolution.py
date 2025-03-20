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

    def adaptive_mutation_factor(self):
        if len(self.performance_memory) < 2:
            return self.mutation_factor
        return 0.5 + 0.1 * np.tanh(self.performance_memory[-1] - self.performance_memory[-2])

    def adaptive_crossover_rate(self):
        if len(self.performance_memory) < 2:
            return self.crossover_rate
        return 0.7 + 0.2 * np.tanh(np.std(self.performance_memory[-5:]))

    def select_parents(self, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        return np.random.choice(indices, 3, replace=False)

    def mutate_and_crossover(self, target_idx, func):
        a, b, c = self.select_parents(target_idx)
        mutant_vector = self.population[a] + self.adaptive_mutation_factor() * (self.population[b] - self.population[c])
        mutant_vector = np.clip(mutant_vector, func.bounds.lb, func.bounds.ub)
        trial_vector = np.copy(self.population[target_idx])
        crossover_points = np.random.rand(self.dim) < self.adaptive_crossover_rate()
        trial_vector[crossover_points] = mutant_vector[crossover_points]
        return trial_vector

    def local_search(self, individual, func):
        perturbation = np.random.normal(0, 0.1, size=self.dim)
        new_individual = np.clip(individual + perturbation, func.bounds.lb, func.bounds.ub)
        return new_individual if func(new_individual) < func(individual) else individual

    def optimize(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        self.current_evaluations += self.population_size
        for _ in range(self.budget - self.current_evaluations):
            for i in range(self.population_size):
                trial_vector = self.mutate_and_crossover(i, func)
                trial_vector = self.local_search(trial_vector, func)
                trial_fitness = func(trial_vector)
                self.current_evaluations += 1
                if trial_fitness < fitness[i]:
                    self.population[i] = trial_vector
                    fitness[i] = trial_fitness
            self.performance_memory.append(np.min(fitness))
        return self.population[np.argmin(fitness)]

    def __call__(self, func):
        return self.optimize(func)