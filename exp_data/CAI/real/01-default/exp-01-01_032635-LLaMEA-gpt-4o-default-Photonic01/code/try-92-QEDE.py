import numpy as np

class QEDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.quantum_factor = 0.05
        self.population = None
        self.best_solution = None
        self.best_value = np.inf

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def quantum_perturbation(self, vector):
        perturbation = np.random.normal(0, self.quantum_factor, self.dim)
        return vector + perturbation

    def select_parents(self, idx):
        candidates = list(range(self.population_size))
        candidates.remove(idx)
        parents = np.random.choice(candidates, 3, replace=False)
        return parents

    def create_offspring(self, idx, lb, ub):
        a, b, c = self.select_parents(idx)
        mutant_vector = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
        mutant_vector = np.clip(mutant_vector, lb, ub)

        trial_vector = np.copy(self.population[idx])
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        trial_vector[crossover_mask] = mutant_vector[crossover_mask]
        
        quantum_trial = self.quantum_perturbation(trial_vector)
        return np.clip(quantum_trial, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                trial_vector = self.create_offspring(i, lb, ub)
                trial_value = func(trial_vector)
                evaluations += 1

                current_value = func(self.population[i])
                evaluations += 1

                if trial_value < current_value:
                    self.population[i] = trial_vector
                    current_value = trial_value

                if current_value < self.best_value:
                    self.best_value = current_value
                    self.best_solution = self.population[i].copy()

        return self.best_solution, self.best_value