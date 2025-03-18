import numpy as np

class AdaptiveDifferentialLearning:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(4 + int(3 * np.log(self.dim)), 50)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        
    def __call__(self, func):
        bounds = func.bounds
        lower_bound, upper_bound = bounds.lb, bounds.ub
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # Differential Mutation
                mutant = np.clip(a + self.mutation_factor * (b - c), lower_bound, upper_bound)
                
                # Crossover
                crossover = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness
            
            # Adaptive learning to adjust mutation factor and crossover rate
            fitness_std = np.std(fitness)
            self.mutation_factor = 0.5 + 0.5 * (1 - fitness_std)
            self.crossover_rate = 0.6 + 0.4 * np.random.rand()
        
        best_index = np.argmin(fitness)
        return population[best_index]