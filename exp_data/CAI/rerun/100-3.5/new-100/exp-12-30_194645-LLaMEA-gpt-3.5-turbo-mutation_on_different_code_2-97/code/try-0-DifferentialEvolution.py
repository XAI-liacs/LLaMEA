import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        def clip(x):
            return np.clip(x, func.bounds.lb, func.bounds.ub)

        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        for _ in range(self.budget):
            for i in range(self.pop_size):
                indices = np.random.choice(self.pop_size, size=3, replace=False)
                target_vector = clip(population[indices[0]] + self.F * (population[indices[1]] - population[indices[2]]))
                trial_vector = np.where(np.random.rand(self.dim) < self.CR, target_vector, population[i])
                trial_vector_fitness = func(trial_vector)
                if trial_vector_fitness < func(population[i]):
                    population[i] = trial_vector
                    if trial_vector_fitness < self.f_opt:
                        self.f_opt = trial_vector_fitness
                        self.x_opt = trial_vector

        return self.f_opt, self.x_opt