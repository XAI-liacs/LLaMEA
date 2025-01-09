import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.pop_size
        
        F_base, CR_base = 0.5, 0.9  # Base mutation and crossover rates
        while evals < self.budget:
            F = F_base * (1 + np.random.rand())  # Adaptive scaling factor
            CR = CR_base * (1 + np.random.rand())  # Adaptive crossover probability
            
            if evals % (self.budget // 10) == 0:  # Adaptive population size adjustment every 10% of budget
                self.pop_size = max(5, int(self.pop_size * 0.9))  # Reduce pop size by 10%, min 5

            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                mutant = np.clip(a + F * (b - c), bounds[0], bounds[1])
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[i])
                
                f = func(trial)
                evals += 1
                if f < fitness[i]:
                    fitness[i] = f
                    population[i] = trial
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial

                if evals >= self.budget:
                    break
        
        return self.f_opt, self.x_opt