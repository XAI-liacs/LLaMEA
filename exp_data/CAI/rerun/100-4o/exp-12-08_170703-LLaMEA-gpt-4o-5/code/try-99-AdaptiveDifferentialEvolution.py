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
        self.success_CR = []
        self.success_F = []

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
                        self.success_CR.append(self.CR)
                        self.success_F.append(self.F)
                        
                # Adapt parameters based on success
                if f_trial < self.f_opt:
                    self.CR = np.mean(self.success_CR[-5:]) if self.success_CR else 0.5 * (1 + np.random.rand()) 
                    self.F = np.mean(self.success_F[-5:]) if self.success_F else 0.5 * (1 + np.random.rand())

        return self.f_opt, self.x_opt