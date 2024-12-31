import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, CR=0.9, F=0.8):
        self.budget = budget
        self.dim = dim
        self.CR = CR
        self.F = F
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = min(10 * self.dim, 100)
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(pop_size, self.dim))
        
        for i in range(self.budget):
            new_population = []
            for j in range(pop_size):
                idxs = np.random.choice(pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs]

                mutant = population[j] + self.F * (x_r1 - population[j]) + self.F * (x_r2 - x_r3)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[j])
                
                f_trial = func(trial)
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                
                new_population.append(trial if f_trial < func(population[j]) else population[j])
            
            population = np.array(new_population)

        return self.f_opt, self.x_opt