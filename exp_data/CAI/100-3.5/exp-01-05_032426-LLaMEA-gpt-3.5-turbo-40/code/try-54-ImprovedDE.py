import numpy as np

class ImprovedDE(DifferentialEvolution):
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_lb=0.2, F_ub=0.8, CR_lb=0.3, CR_ub=0.9):
        super().__init__(budget, dim, F, CR)
        self.F_lb = F_lb
        self.F_ub = F_ub
        self.CR_lb = CR_lb
        self.CR_ub = CR_ub

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            F_i = np.random.uniform(self.F_lb, self.F_ub)
            CR_i = np.random.uniform(self.CR_lb, self.CR_ub)

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