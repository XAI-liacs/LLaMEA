import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=20):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)
    
    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        # Update the best found solution
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Parameter adaptation
                F = 0.5 + np.random.rand() * 0.3  # Adaptive scale factor
                CR = 0.5 + np.random.rand() * 0.3  # Adaptive crossover rate

                # Mutation
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])

                # Crossover
                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

                if evaluations >= self.budget:
                    break
        
        return self.f_opt, self.x_opt