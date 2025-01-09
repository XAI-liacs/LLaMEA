import numpy as np

class OppositionalAdaptiveDE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20
        self.bounds = [-5.0, 5.0]

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx].copy()
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            # Opposition-based learning
            opposite_population = self.bounds[0] + self.bounds[1] - population
            opposite_fitness = np.array([func(ind) for ind in opposite_population])
            evaluations += self.population_size
            
            for i in range(self.population_size):
                if opposite_fitness[i] < fitness[i]:
                    population[i], fitness[i] = opposite_population[i], opposite_fitness[i]
            
            # Adaptive Differential Evolution
            F = 0.5 + np.random.rand() * 0.5  # Mutation factor
            CR = 0.5 + np.random.rand() * 0.3  # Crossover rate
            
            for i in range(self.population_size):
                # Mutation and Crossover
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])
                
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()
                
                if evaluations >= self.budget:
                    break
        
        return self.f_opt, self.x_opt