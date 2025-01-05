import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __mutation(self, population, target_idx):
        r1, r2, r3 = np.random.choice(len(population), 3, replace=False)
        mutant_vector = population[r1] + self.F * (population[r2] - population[r3])
        return mutant_vector

    def __crossover(self, target_vector, mutant_vector):
        crossover_mask = np.random.rand(self.dim) < self.CR
        trial_vector = np.where(crossover_mask, mutant_vector, target_vector)
        return trial_vector

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            for j in range(len(population)):
                target_vector = population[j]
                mutant_vector = self.__mutation(population, j)
                trial_vector = self.__crossover(target_vector, mutant_vector)
                
                f = func(trial_vector)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial_vector
        
        return self.f_opt, self.x_opt