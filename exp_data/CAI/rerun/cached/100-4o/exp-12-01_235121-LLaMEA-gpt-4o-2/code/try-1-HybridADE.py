import numpy as np

class HybridADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + int(0.5 * self.dim)
        self.F = 0.5   # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.evaluations = 0
    
    def __call__(self, func):
        best_solution = None
        best_fitness = float('inf')
        fitness = np.array([func(ind) for ind in self.population])
        self.evaluations += self.population_size
        
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                
                # Mutation
                indices = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
                x1, x2, x3 = self.population[indices]
                mutant = np.clip(x1 + self.F * (x2 - x3), self.lower_bound, self.upper_bound)
                
                # Crossover
                jrand = np.random.randint(self.dim)
                trial = np.array([mutant[j] if (np.random.rand() < self.CR or j == jrand) else self.population[i, j] for j in range(self.dim)])
                
                # Selection
                trial_fitness = func(trial)
                self.evaluations += 1
                
                if trial_fitness < fitness[i]:
                    self.population[i] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial

            # Dynamic population size adjustment
            if self.evaluations < self.budget * 0.8:
                self.population_size = 10 + int(0.5 * self.dim) + int(0.2 * self.dim * (self.evaluations / self.budget))
            else:
                self.population_size = max(5, int(self.population_size * 0.9))

        return best_solution