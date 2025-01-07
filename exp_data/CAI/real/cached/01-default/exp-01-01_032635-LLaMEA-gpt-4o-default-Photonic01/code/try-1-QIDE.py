import numpy as np

class QIDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, int(budget / 5))
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_value = np.inf
        self.iteration = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def quantum_superposition(self, target, best, r1, r2):
        superposed_state = target + np.random.uniform(-1, 1, self.dim) * (
            np.abs(best - r1) * np.abs(r2 - best)
        )
        return superposed_state

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)

        while self.iteration < self.budget:
            F = 0.5 + np.random.rand() * 0.5  # Dynamic scaling factor
            CR = 0.1 + np.random.rand() * 0.9  # Dynamic crossover rate

            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                r1, r2, r3 = self.population[np.random.choice(indices, 3, replace=False)]

                mutant = r1 + F * (r2 - r3)
                mutant = np.clip(mutant, lb, ub)

                trial = np.where(np.random.rand(self.dim) < CR, mutant, self.population[i])
                trial = self.quantum_superposition(trial, self.best_solution or self.population[i], r1, r2)
                trial = np.clip(trial, lb, ub)

                trial_value = func(trial)

                if trial_value < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_value

                if trial_value < self.best_value:
                    self.best_solution = trial
                    self.best_value = trial_value

            self.iteration += self.population_size

        return self.best_solution, self.best_value