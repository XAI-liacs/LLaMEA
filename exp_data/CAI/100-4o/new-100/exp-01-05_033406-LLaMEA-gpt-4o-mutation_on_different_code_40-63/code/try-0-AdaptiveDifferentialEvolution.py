import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=20):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = None
        self.bounds = (-5.0, 5.0)
        self.F = 0.5  # Initial scaling factor
        self.CR = 0.7  # Initial crossover rate

    def __call__(self, func):
        # Initialize population
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], 
                                            (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in self.population])
        
        # Evaluate initial population
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = self.population[best_idx]
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])
                
                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    self.population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

            # Adaptive parameter control
            self.F = np.random.uniform(0.5, 1.0)
            self.CR = np.random.uniform(0.1, 0.9)

        return self.f_opt, self.x_opt