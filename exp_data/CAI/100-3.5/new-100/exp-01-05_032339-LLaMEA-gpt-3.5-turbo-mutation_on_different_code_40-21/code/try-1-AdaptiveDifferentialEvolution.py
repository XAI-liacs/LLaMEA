import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F_min=0.4, F_max=0.9, CR_min=0.7, CR_max=0.9):
        self.budget = budget
        self.dim = dim
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max
        self.f_opt = np.Inf
        self.x_opt = None

    def __mutation(self, population, target_idx):
        r1, r2, r3 = np.random.choice(len(population), 3, replace=False)
        F = np.random.uniform(self.F_min, self.F_max)
        mutant_vector = population[r1] + F * (population[r2] - population[r3])
        return mutant_vector

    def __crossover(self, target_vector, mutant_vector):
        CR = np.random.uniform(self.CR_min, self.CR_max)
        crossover_mask = np.random.rand(self.dim) < CR
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