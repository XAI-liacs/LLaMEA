import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F_init=0.5, CR_init=0.9, F_scaling=0.9, CR_scaling=0.1):
        self.budget = budget
        self.dim = dim
        self.F = F_init  # Initialize F
        self.CR = CR_init  # Initialize CR
        self.F_init = F_init  # Store initial F
        self.CR_init = CR_init  # Store initial CR
        self.F_scaling = F_scaling  # Scaling factor for F
        self.CR_scaling = CR_scaling  # Scaling factor for CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            F_i = np.clip(self.F_init * np.exp(self.F_scaling * i / self.budget), 0, 2)  # Adapt F
            CR_i = np.clip(self.CR_init * np.exp(self.CR_scaling * i / self.budget), 0, 1)  # Adapt CR

            mutant = population[a] + F_i * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < CR_i
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring

            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

        return self.f_opt, self.x_opt