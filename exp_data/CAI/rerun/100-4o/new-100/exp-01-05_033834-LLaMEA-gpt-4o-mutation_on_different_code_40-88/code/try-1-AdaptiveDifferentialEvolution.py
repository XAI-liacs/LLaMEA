import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for evals in range(self.population_size, self.budget):
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 5, replace=False)
                x1, x2, x3, x4, x5 = population[indices]
                
                if np.random.rand() < 0.5:
                    mutant = x1 + self.mutation_factor * (x2 - x3) + self.mutation_factor * (x4 - x5)
                else:
                    mutant = x1 + self.mutation_factor * (x2 - x3)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Ensure bounds
                trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
                
                # Selection
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        self.local_search(func, trial)  # Local search invoked

            # Adaptive parameters update
            self.mutation_factor = np.clip(self.mutation_factor * (1 + 0.1 * (np.random.rand() - 0.5)), 0.1, 0.9)
            self.crossover_rate = np.clip(self.crossover_rate * (1 + 0.1 * (np.random.rand() - 0.5)), 0.1, 0.9)
        
        return self.f_opt, self.x_opt

    def local_search(self, func, x):
        step_size = 0.1
        for _ in range(5):
            candidate = x + np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(candidate, func.bounds.lb, func.bounds.ub)
            f_candidate = func(candidate)
            if f_candidate < self.f_opt:
                self.f_opt = f_candidate
                self.x_opt = candidate