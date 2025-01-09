import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, population_size=20):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None

    def mutate(self, population, target_idx):
        candidates = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        return population[a] + self.F * (population[b] - population[c])

    def crossover(self, target_vector, trial_vector):
        crossover_points = np.random.rand(self.dim) < self.CR
        new_vector = np.where(crossover_points, trial_vector, target_vector)
        return new_vector

    def select(self, func, target_vector, trial_vector):
        f_target = func(target_vector)
        f_trial = func(trial_vector)

        if f_trial < f_target:
            return f_trial, trial_vector
        else:
            return f_target, target_vector

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.population_size, self.dim))

        for i in range(self.budget):
            for j in range(self.population_size):
                target_vector = population[j]
                trial_vector = self.mutate(population, j)
                new_vector = self.crossover(target_vector, trial_vector)
                f_opt, x_opt = self.select(func, target_vector, new_vector)

                if f_opt < self.f_opt:
                    self.f_opt = f_opt
                    self.x_opt = x_opt

        return self.f_opt, self.x_opt