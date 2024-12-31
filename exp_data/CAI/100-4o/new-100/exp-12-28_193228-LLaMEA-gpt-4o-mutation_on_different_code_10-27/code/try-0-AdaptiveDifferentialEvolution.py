import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=20):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        for i in range(self.population_size):
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i]
        
        evaluations = self.population_size

        while evaluations < self.budget:
            for j in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                # Select indices for mutation
                indices = list(range(self.population_size))
                indices.remove(j)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                # Mutation and crossover
                F = np.random.uniform(0.5, 1.0)  # Mutation factor
                CR = np.random.uniform(0.0, 1.0)  # Crossover probability
                mutant = np.clip(pop[a] + F * (pop[b] - pop[c]), lb, ub)
                trial = np.where(np.random.rand(self.dim) < CR, mutant, pop[j])
                
                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[j]:
                    fitness[j] = f_trial
                    pop[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

        return self.f_opt, self.x_opt