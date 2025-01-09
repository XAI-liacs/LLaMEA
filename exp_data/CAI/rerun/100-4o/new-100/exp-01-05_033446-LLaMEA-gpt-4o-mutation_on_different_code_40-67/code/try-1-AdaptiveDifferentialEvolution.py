import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=100):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)
        self.scale_factors = np.array([0.5, 0.8, 1.2])
        self.cr = 0.9
        self.sf_prob = np.ones(3) / 3

    def mutation(self, target_idx, population, scale_factor):
        indices = list(range(len(population)))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = population[a] + scale_factor * (population[b] - population[c])
        return np.clip(mutant, *self.bounds)
    
    def crossover(self, target, mutant, cr):
        cross_points = np.random.rand(self.dim) < cr
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def __call__(self, func):
        population = np.random.uniform(*self.bounds, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= self.pop_size
        
        while self.budget > 0:
            improvements = np.zeros(3)
            for i in range(self.pop_size):
                sf_idx = np.random.choice(len(self.scale_factors), p=self.sf_prob)
                mutate_sf = self.scale_factors[sf_idx]
                cr = np.random.normal(self.cr, 0.1)
                mutant = self.mutation(i, population, mutate_sf)
                trial = self.crossover(population[i], mutant, cr)
                
                f_trial = func(trial)
                self.budget -= 1
                
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    improvements[sf_idx] += 1
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if self.budget <= 0:
                    break
            
            if np.sum(improvements) > 0:
                self.sf_prob = improvements / np.sum(improvements)
        
        return self.f_opt, self.x_opt