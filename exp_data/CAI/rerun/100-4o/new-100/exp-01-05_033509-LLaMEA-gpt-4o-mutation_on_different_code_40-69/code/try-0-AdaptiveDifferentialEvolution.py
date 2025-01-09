import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        if np.min(fitness) < self.f_opt:
            self.f_opt = np.min(fitness)
            self.x_opt = population[np.argmin(fitness)]

        eval_count = self.population_size
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    
                a, b, c = population[indices]
                mutant = np.clip(a + self.mutation_factor * (b - c), bounds[0], bounds[1])
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                f_trial = func(trial)
                eval_count += 1

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if eval_count >= self.budget:
                    break
        
        return self.f_opt, self.x_opt