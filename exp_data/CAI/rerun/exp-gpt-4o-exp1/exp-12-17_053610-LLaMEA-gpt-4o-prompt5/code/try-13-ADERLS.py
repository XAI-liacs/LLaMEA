import numpy as np

class ADERLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.best_solution = None
        self.best_score = float('inf')
        self.momentum = np.zeros(dim)  # Initialize momentum term
        
    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in population])
        self.best_solution = population[np.argmin(scores)]
        self.best_score = np.min(scores)
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # Dynamic mutation factor
                self.F = 0.3 + 0.4 * np.random.rand()
                # Slight increase in the bias towards the global best
                mutant = np.clip(a + self.F * (b - c) + 0.20 * (self.best_solution - a) + 0.1 * self.momentum, 
                                 self.lower_bound, self.upper_bound)  # Added momentum influence
                
                # Dynamic crossover probability
                self.CR = 0.8 + 0.2 * np.random.rand()
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                
                # Adaptive Randomized Local Search probability
                if np.random.rand() < (0.1 + 0.1 * (evaluations / self.budget) * (1 + 0.05 * np.random.rand())):
                    trial = trial + np.random.normal(0, 0.1, self.dim)
                    trial = np.clip(trial, self.lower_bound, self.upper_bound)
                
                score = func(trial)
                evaluations += 1
                
                if score < scores[i]:
                    self.momentum = trial - population[i]  # Update momentum with successful movement
                    population[i] = trial
                    scores[i] = score
                    
                    if score < self.best_score:
                        self.best_solution = trial
                        self.best_score = score
        
        return self.best_solution