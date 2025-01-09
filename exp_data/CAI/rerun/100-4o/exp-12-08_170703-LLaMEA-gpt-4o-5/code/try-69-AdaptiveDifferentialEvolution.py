import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # A common heuristic for DE
        self.f_opt = np.Inf
        self.x_opt = None
        self.CR = 0.9  # Crossover probability
        self.F = 0.8  # Differential weight
        self.bounds = (-5.0, 5.0)
        
    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget // self.population_size):
            success_count = 0
            for i in range(self.population_size):
                # Mutation: select three distinct individuals
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])
                
                # Crossover
                trial = np.array([mutant[j] if np.random.rand() < self.CR else population[i][j] 
                                  for j in range(self.dim)])
                
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    success_count += 1  # Track successful trials
                    
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        
            # Adapt parameters based on success
            if success_count > self.population_size / 5:  # Adjust only if success is above a threshold
                self.F = min(0.9, self.F * 1.1)  # Increase F slightly if successful
                self.CR = min(0.9, self.CR * 1.1)  # Increase CR slightly if successful

        return self.f_opt, self.x_opt