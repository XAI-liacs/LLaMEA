import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
    
    def mutate(self, population, target_idx):
        candidates = [idx for idx in range(len(population)) if idx != target_idx]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = population[a] + self.F * (population[b] - population[c])
        return mutant
    
    def crossover(self, target, mutant):
        crossover_points = np.random.rand(self.dim) < self.CR
        trial = np.where(crossover_points, mutant, target)
        return trial
    
    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        
        for i in range(self.budget):
            for j in range(self.pop_size):
                target = population[j]
                mutant = self.mutate(population, j)
                trial = self.crossover(target, mutant)
                
                f_target = func(target)
                f_trial = func(trial)
                
                if f_trial < f_target:
                    population[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    if f_target < self.f_opt:
                        self.f_opt = f_target
                        self.x_opt = target
        
        return self.f_opt, self.x_opt