import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Rule of thumb
        self.f_opt = np.Inf
        self.x_opt = None
        self.mutation_factor = 0.5
        self.crossover_rate = 0.5
        self.bounds = (-5.0, 5.0)
        
    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], 
                                       (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                # Select three random indices distinct from i
                indices = np.delete(np.arange(self.population_size), i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Mutate
                mutant = np.clip(a + self.mutation_factor * (b - c), self.bounds[0], self.bounds[1])

                # Crossover
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    # Update the best found
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

                # Adaptation of mutation factor and crossover rate
                self.mutation_factor = 0.4 + 0.1 * np.random.rand()
                self.crossover_rate = 0.3 + 0.4 * np.random.rand()
                
        return self.f_opt, self.x_opt