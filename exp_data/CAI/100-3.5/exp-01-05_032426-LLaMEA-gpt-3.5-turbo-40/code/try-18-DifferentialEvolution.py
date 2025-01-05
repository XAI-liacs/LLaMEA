import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None
        self.F_history = [F] * budget
        self.CR_history = [CR] * budget

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            F_val = np.mean(self.F_history)
            CR_val = np.mean(self.CR_history)

            mutant = population[a] + F_val * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < CR_val
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring
            
                self.F_history[i] = self.F * np.random.normal(1, 0.1)
                self.CR_history[i] = self.CR * np.random.normal(1, 0.1)
            
            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

        return self.f_opt, self.x_opt