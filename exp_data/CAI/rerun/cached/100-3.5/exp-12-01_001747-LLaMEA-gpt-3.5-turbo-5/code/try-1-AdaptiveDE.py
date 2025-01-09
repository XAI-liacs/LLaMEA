import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.cr = 0.5
        self.f = 0.5
        self.max_iter = budget // self.pop_size
        
    def __call__(self, func):
        def mutate(x, pop, f):
            a, b, c = np.random.choice(pop, 3, replace=False)
            return np.clip(a + f * (b - c), -5, 5)
        
        def crossover(x, mutant, cr):
            mask = np.random.rand(self.dim) < cr
            trial = np.where(mask, mutant, x)
            return trial
        
        pop = np.random.uniform(-5, 5, (self.pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        
        for _ in range(self.max_iter):
            new_pop = np.zeros((self.pop_size, self.dim))
            for i in range(self.pop_size):
                mutant = mutate(pop[i], pop, self.f)
                trial = crossover(pop[i], mutant, self.cr)
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    new_pop[i] = trial
                else:
                    new_pop[i] = pop[i]
                    
            pop = new_pop
            fitness = np.array([func(x) for x in pop])
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < func(best):
                best = pop[best_idx]
                
        return best