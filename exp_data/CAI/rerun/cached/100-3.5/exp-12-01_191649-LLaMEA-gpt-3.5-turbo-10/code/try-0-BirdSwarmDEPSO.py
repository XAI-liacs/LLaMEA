import numpy as np

class BirdSwarmDEPSO:
    def __init__(self, budget, dim, pop_size=20, c1=2.0, c2=2.0, f=0.5, cr=0.7):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.c1 = c1
        self.c2 = c2
        self.f = f
        self.cr = cr

    def __call__(self, func):
        lb = -5.0
        ub = 5.0
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        for _ in range(self.budget - self.pop_size):
            for i in range(self.pop_size):
                r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                mutant = pop[r1] + self.f * (pop[r2] - pop[r3])
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, pop[i])
                fitness_trial = func(trial)
                
                if fitness_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = fitness_trial
            
            g_best_idx = np.argmin(fitness)
            g_best = pop[g_best_idx]
            
            for i in range(self.pop_size):
                velocity = self.c1 * np.random.rand(self.dim) * (pop[i] - pop[i]) + \
                           self.c2 * np.random.rand(self.dim) * (g_best - pop[i])
                pop[i] += velocity
                pop[i] = np.clip(pop[i], lb, ub)
                fitness[i] = func(pop[i])
        
        return pop[np.argmin(fitness)]