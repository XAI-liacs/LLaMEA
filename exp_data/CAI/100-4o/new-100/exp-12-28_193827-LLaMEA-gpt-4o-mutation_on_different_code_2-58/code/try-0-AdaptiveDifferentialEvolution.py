import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.inf
        self.x_opt = None
        self.F = 0.8  # Mutation factor
        self.CR = 0.9  # Crossover rate
        self.success_rate = 0.1  # Initial success rate

    def __call__(self, func):
        bounds = func.bounds
        population = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for evals in range(self.pop_size, self.budget, self.pop_size):
            new_population = []
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                mutant = np.clip(a + self.F * (b - c), bounds.lb, bounds.ub)
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[i])
                
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    new_population.append(trial)
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        self.success_rate = min(1.0, self.success_rate * 1.05)
                else:
                    new_population.append(population[i])
                    self.success_rate = max(0.1, self.success_rate * 0.95)
            
            self.F = 0.5 + 0.3 * np.random.rand() * self.success_rate
            self.CR = 0.7 + 0.2 * np.random.rand() * self.success_rate
            population = np.array(new_population)
        
        return self.f_opt, self.x_opt