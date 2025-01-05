import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 100
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        self.initialize_population(func.bounds.lb, func.bounds.ub)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                # Mutation
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = self.population[a] + self.F * (self.population[b] - self.population[c])
                mutant_vector = np.clip(mutant_vector, func.bounds.lb, func.bounds.ub)
                
                # Crossover
                trial_vector = np.copy(self.population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        trial_vector[j] = mutant_vector[j]
                
                # Selection
                trial_fitness = func(trial_vector)
                evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness

                # Update best solution found
                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial_vector

        return self.f_opt, self.x_opt

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            if self.fitness[i] < self.f_opt:
                self.f_opt = self.fitness[i]
                self.x_opt = self.population[i]