import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= self.pop_size
        
        while self.budget > 0:
            F = np.random.normal(0.5, 0.3)
            F = np.clip(F, 0, 1)
            CR = np.random.normal(0.5, 0.1)
            CR = np.clip(CR, 0, 1)
            
            for i in range(self.pop_size):
                if self.budget <= 0:
                    break
                
                indices = np.arange(self.pop_size)
                indices = np.delete(indices, i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, population[i])
                
                f = func(trial)
                self.budget -= 1
                
                if f < fitness[i]:
                    fitness[i] = f
                    population[i] = trial
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial
                        
            # Elitism: Preserve best individuals
            elite_indices = np.argsort(fitness)[:int(self.pop_size * 0.1)]
            elite_population = population[elite_indices]
            population = np.append(population, elite_population, axis=0)
            self.pop_size = len(population)

        return self.f_opt, self.x_opt