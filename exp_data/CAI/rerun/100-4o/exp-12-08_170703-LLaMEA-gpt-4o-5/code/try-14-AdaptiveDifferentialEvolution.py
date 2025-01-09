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
                    
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        
                # Adapt parameters based on success
                if f_trial < self.f_opt:
                    self.F = 0.5 * (1 + np.random.rand())  # Dynamic adjustment
                    self.CR = 0.5 * (1 + np.random.rand())

            # Check population diversity to adjust F
            diversity = np.std(fitness)
            if diversity < 1e-5:  # If diversity is very low, increase mutation factor
                self.F = min(1.5, self.F * 1.2)

        return self.f_opt, self.x_opt