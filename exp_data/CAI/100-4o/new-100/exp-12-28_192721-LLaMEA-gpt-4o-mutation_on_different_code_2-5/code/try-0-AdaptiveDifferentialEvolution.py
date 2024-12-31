import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size
        
        while budget_used < self.budget:
            for i in range(self.population_size):
                if budget_used >= self.budget:
                    break
                    
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                
                mutant = np.clip(a + self.mutation_factor * (b - c), bounds[0], bounds[1])
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                f = func(trial)
                budget_used += 1
                
                if f < fitness[i]:
                    population[i] = trial
                    fitness[i] = f
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial

            # Dynamically adjust mutation factor and crossover rate
            self.mutation_factor = 0.5 + np.random.rand() * 0.5
            self.crossover_rate = 0.7 + np.random.rand() * 0.3
        
        return self.f_opt, self.x_opt