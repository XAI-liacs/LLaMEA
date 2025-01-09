import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        
        evals = self.pop_size
        F_values = np.full(self.pop_size, self.F)
        CR_values = np.full(self.pop_size, self.CR)

        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # Adaptive Mutation Strategy
                F_adaptive = F_values[i] * np.random.uniform(0.5, 1.5)
                mutant = np.clip(a + F_adaptive * (b - c), lb, ub)
                
                # Adaptive Crossover Rate
                CR_adaptive = CR_values[i] * np.random.uniform(0.8, 1.2)
                crossover_mask = np.random.rand(self.dim) < CR_adaptive
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Evaluate Trial
                f_trial = func(trial)
                evals += 1

                # Self-adaptive parameter adjustment
                if f_trial < fitness[i]:
                    F_values[i] = F_adaptive
                    CR_values[i] = CR_adaptive
                    
                # Selection
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
        
        return self.f_opt, self.x_opt