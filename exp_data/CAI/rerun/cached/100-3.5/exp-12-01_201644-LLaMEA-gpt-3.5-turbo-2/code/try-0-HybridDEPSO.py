import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.max_iter = budget // self.pop_size
        self.cr = 0.8
        self.f = 0.5
        self.w = 0.5

    def de(self, target, pop, f):
        r1, r2, r3 = np.random.choice(len(pop), 3, replace=False)
        mutant = pop[r1] + f * (pop[r2] - pop[r3])
        crossover = np.random.rand(self.dim) < self.cr
        trial = np.where(crossover, mutant, target)
        return trial

    def pso(self, target, pop, gbest):
        inertia_weight = self.w
        cognitive_weight = 1.5
        social_weight = 1.5
        velocity = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        
        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                r1, r2 = np.random.random((2, self.dim))
                velocity[i] = (inertia_weight * velocity[i] + 
                               cognitive_weight * r1 * (gbest - pop[i]) + 
                               social_weight * r2 * (target - pop[i]))
                pop[i] = np.clip(pop[i] + velocity[i], -5.0, 5.0)
        return pop

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        gbest = np.copy(pop[np.argmin([func(ind) for ind in pop])])
        
        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                trial = self.de(pop[i], pop, self.f)
                trial_fit = func(trial)
                if trial_fit < func(pop[i]):
                    pop[i] = trial
            
            pop = self.pso(pop[np.argmin([func(ind) for ind in pop])], pop, gbest)
            gbest = np.copy(pop[np.argmin([func(ind) for ind in pop])])
        
        return gbest