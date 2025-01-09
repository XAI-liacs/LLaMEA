import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, f=0.8, cr=0.9, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.f = f
        self.cr = cr
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def mutate(self, population, target_idx):
        candidates = np.random.choice(population, size=3, replace=False)
        mutant_vector = candidates[0] + self.f * (candidates[1] - candidates[2])
        return mutant_vector

    def crossover(self, target_vector, mutant_vector):
        crossover_mask = np.random.rand(self.dim) < self.cr
        trial_vector = np.where(crossover_mask, mutant_vector, target_vector)
        return trial_vector

    def select(self, func, target_vector, trial_vector):
        f_target = func(target_vector)
        f_trial = func(trial_vector)

        if f_trial < f_target:
            return trial_vector, f_trial
        else:
            return target_vector, f_target

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        
        for i in range(self.budget):
            for j in range(self.pop_size):
                target_vector = population[j]
                
                mutant_vector = self.mutate(population, j)
                trial_vector = self.crossover(target_vector, mutant_vector)
                
                population[j], f = self.select(func, target_vector, trial_vector)
                
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = population[j]
            
        return self.f_opt, self.x_opt