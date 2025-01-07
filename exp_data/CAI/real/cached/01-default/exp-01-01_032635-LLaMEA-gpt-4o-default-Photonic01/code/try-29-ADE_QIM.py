import numpy as np

class ADE_QIM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget)
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.population = None
        self.bounds = None

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        self.bounds = (lb, ub)

    def differential_mutation(self, target_idx):
        indices = list(range(self.pop_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
        lb, ub = self.bounds
        return np.clip(mutant, lb, ub)

    def quantum_inspired_mutation(self, individual):
        global_best = self.population[np.argmin([func(ind) for ind in self.population])]
        beta = np.random.normal(0, 1, self.dim)
        new_position = individual + beta * (global_best - individual)
        lb, ub = self.bounds
        return np.clip(new_position, lb, ub)

    def crossover(self, target, mutant):
        trial = np.copy(target)
        crossover_points = np.random.rand(self.dim) < self.crossover_rate
        trial[crossover_points] = mutant[crossover_points]
        return trial

    def select(self, target, trial, target_idx, func):
        target_value = func(target)
        trial_value = func(trial)
        if trial_value < target_value:
            self.population[target_idx] = trial
            return trial_value
        else:
            return target_value

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0
        best_value = np.inf
        best_position = None

        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                mutant = self.differential_mutation(i)
                if np.random.rand() < 0.1:
                    mutant = self.quantum_inspired_mutation(mutant)

                trial = self.crossover(self.population[i], mutant)
                target_value = self.select(self.population[i], trial, i, func)
                evaluations += 1

                if target_value < best_value:
                    best_value = target_value
                    best_position = trial

        return best_position, best_value