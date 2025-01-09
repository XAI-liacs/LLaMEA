import numpy as np

class HybridEvolutionaryAnnealing:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20
        self.temp = 1.0

    def __call__(self, func):
        # Initialize a population of candidate solutions
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]

                # Adaptive Differential Evolution mutation
                F = 0.5 + np.random.rand() * 0.4  # adaptive mutation factor
                mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < 0.9
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluate fitness of trial
                f_trial = func(trial)
                
                # Enhanced selection strategy
                if f_trial < fitness[i] or np.random.rand() < np.exp((fitness[i] - f_trial) / self.temp):
                    population[i] = trial
                    fitness[i] = f_trial
                    
                    # Update the best solution found
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
            
            # Temperature decay to gradually reduce exploration
            self.temp *= 0.995
        
        return self.f_opt, self.x_opt