import numpy as np

class QEDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, budget)  # Population size for diverse search
        self.f_mutation = 0.8  # Mutation factor
        self.cr_crossover = 0.9  # Crossover rate
        self.best_solution = None
        self.best_value = np.inf

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def quantum_position_update(self, position, best, lb, ub):
        """Quantum-inspired position update."""
        shift = np.random.normal(0, 1, self.dim) * (best - position) / 2
        new_position = position + shift
        return np.clip(new_position, lb, ub)

    def differential_mutation(self, population, best_idx, lb, ub):
        idxs = np.random.choice(population.shape[0], 3, replace=False)
        x1, x2, x3 = population[idxs]
        mutant_vector = x1 + self.f_mutation * (x2 - x3)
        mutant_vector = np.clip(mutant_vector, lb, ub)
        
        if np.random.rand() < 0.5:
            mutant_vector = self.quantum_position_update(mutant_vector, population[best_idx], lb, ub)
        
        return mutant_vector

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.cr_crossover
        trial_vector = np.where(crossover_mask, mutant, target)
        return trial_vector

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        population_values = np.array([func(ind) for ind in population])
        best_idx = np.argmin(population_values)
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                mutant_vector = self.differential_mutation(population, best_idx, lb, ub)
                trial_vector = self.crossover(population[i], mutant_vector)

                trial_value = func(trial_vector)
                evaluations += 1

                if trial_value < population_values[i]:
                    population[i] = trial_vector
                    population_values[i] = trial_value
                    if trial_value < population_values[best_idx]:
                        best_idx = i

            self.best_solution, self.best_value = population[best_idx], population_values[best_idx]
        
        return self.best_solution, self.best_value