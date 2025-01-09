import numpy as np

class HybridDE_SA:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.bounds = [-5.0, 5.0]
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.temperature = 100.0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for iteration in range(self.budget - self.population_size):
            # Dynamically adjust mutation factor
            adaptive_mutation_factor = self.mutation_factor * (1 - iteration / self.budget)
            
            # Differential Evolution mutation and crossover
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + adaptive_mutation_factor * (b - c), self.bounds[0], self.bounds[1])
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])

                # Simulated annealing acceptance
                f_trial = func(trial)
                if f_trial < fitness[i] or np.random.rand() < np.exp((fitness[i] - f_trial) / self.temperature):
                    population[i] = trial
                    fitness[i] = f_trial

                # Update best solution
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

            # Annealing schedule (cooling)
            self.temperature *= 0.99  # Cooling rate

        return self.f_opt, self.x_opt