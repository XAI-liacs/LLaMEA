import numpy as np

class BCDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.F = 0.8

    def boundary_handling(self, x):
        return np.clip(x, self.lb, self.ub)

    def adaptive_mutation_scaling(self, idx):
        return np.clip(self.F + 0.1 * (idx / self.budget), 0.1, 0.9)

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for _ in range(self.budget):
            idx = np.argsort(fitness)
            best = population[idx[0]]
            self.F = self.adaptive_mutation_scaling(_)

            mutant = self.boundary_handling(best + self.F * (population[idx[1]] - population[idx[2]]))
            trial = np.where(np.random.uniform(0, 1, self.dim) < 0.9, mutant, population[idx[0]])

            trial_fitness = func(trial)
            if trial_fitness < fitness[idx[0]]:
                population[idx[0]] = trial
                fitness[idx[0]] = trial_fitness

        return population[idx[0]]