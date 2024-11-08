import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 50
        self.c1 = 2.05
        self.c2 = 2.05
        self.w = 0.7
        self.f = 0.5
        self.cr = 0.9
        
    def __call__(self, func):
        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        pbest = pop.copy()
        pbest_fitness = np.array([func(ind) for ind in pbest])
        gbest_idx = np.argmin(pbest_fitness)
        gbest = pbest[gbest_idx]
        gbest_fitness = pbest_fitness[gbest_idx]
        
        evaluations = self.pop_size
        
        while evaluations < self.budget:
            # Pre-generate random numbers
            r1, r2 = np.random.rand(2, self.pop_size, self.dim)
            
            # Update velocities and positions (PSO)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (pbest - pop) +
                          self.c2 * r2 * (gbest - pop))
            pop = np.clip(pop + velocities, self.lb, self.ub)
            
            # Evaluate new population
            fitness = np.array([func(ind) for ind in pop])
            evaluations += self.pop_size
            
            # Update pbest and gbest using vectorized operations
            improved = fitness < pbest_fitness
            pbest_fitness[improved] = fitness[improved]
            pbest[improved] = pop[improved]
            min_idx = np.argmin(pbest_fitness)
            
            if pbest_fitness[min_idx] < gbest_fitness:
                gbest = pbest[min_idx]
                gbest_fitness = pbest_fitness[min_idx]
            
            # Differential Evolution step
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                indices = np.random.choice(self.pop_size, 3, replace=False)
                if i in indices:
                    indices = (set(range(self.pop_size)) - set(indices)).pop()
                a, b, c = indices
                mutant = np.clip(pbest[a] + self.f * (pbest[b] - pbest[c]), self.lb, self.ub)
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, pop[i])
                
                trial_fitness = func(trial)
                evaluations += 1
                
                if trial_fitness < pbest_fitness[i]:
                    pbest[i] = trial
                    pbest_fitness[i] = trial_fitness
                    if trial_fitness < gbest_fitness:
                        gbest = trial
                        gbest_fitness = trial_fitness

        return gbest