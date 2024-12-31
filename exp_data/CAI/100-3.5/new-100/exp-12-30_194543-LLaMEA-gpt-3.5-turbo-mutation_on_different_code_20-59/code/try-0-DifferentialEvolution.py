import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 10
        F_min = 0.5
        F_max = 2.0
        CR = 0.9
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (pop_size, self.dim))
        
        for i in range(self.budget):
            for j in range(pop_size):
                target = population[j]
                a, b, c = np.random.choice(population, 3, replace=False)
                F = np.random.uniform(F_min, F_max)
                trial = a + F * (b - c)
                mask = np.random.rand(self.dim) < CR
                offspring = np.where(mask, trial, target)
                
                f = func(offspring)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = offspring
            
        return self.f_opt, self.x_opt