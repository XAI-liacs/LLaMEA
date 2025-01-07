import numpy as np

class QIDE_Refined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cross_prob = 0.9
        self.diff_weight = 0.8
        self.local_weight = 0.5  # New parameter for local search

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def quantum_superposition(self, population, lb, ub, func):
        beta = 0.15 / np.sqrt(self.dim)  # Adjusted beta
        best_solution = population[np.argmin([func(ind) for ind in population])]
        quantum_population = population + beta * (best_solution - population) * np.random.normal(0, 1, (self.population_size, self.dim))
        np.clip(quantum_population, lb, ub, out=quantum_population)
        return quantum_population

    def adaptive_search(self, population, lb, ub, func):
        new_population = np.copy(population)
        for i in range(self.population_size):
            local_indices = np.random.choice(range(self.population_size), 2, replace=False)
            local_search = population[local_indices[0]] + self.local_weight * (population[local_indices[1]] - population[local_indices[0]])
            global_indices = [idx for idx in range(self.population_size) if idx not in local_indices]
            global_search = population[global_indices[np.argmin([func(population[idx]) for idx in global_indices])]]
            combined_search = (local_search + global_search) / 2  # Combine local and global
            trial = np.clip(combined_search, lb, ub)
            if func(trial) < func(population[i]):
                new_population[i] = trial
        return new_population

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            quantum_population = self.quantum_superposition(population, lb, ub, func)
            population = self.adaptive_search(quantum_population, lb, ub, func)
            evaluations += self.population_size

        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]