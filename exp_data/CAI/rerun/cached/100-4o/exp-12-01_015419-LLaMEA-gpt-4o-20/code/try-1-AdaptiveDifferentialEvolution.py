import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.current_evaluations = 0
        self.F = 0.5  # Differential weight
        self.CR = 0.9 # Crossover probability
    
    def __call__(self, func):
        while self.current_evaluations < self.budget:
            # Evaluate initial population
            for i in range(self.population_size):
                if self.fitness[i] == np.inf:
                    self.fitness[i] = func(self.population[i])
                    self.current_evaluations += 1
                    if self.current_evaluations >= self.budget:
                        return self.population[np.argmin(self.fitness)]

            # Adaptive DE process
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = self.population[indices]
                mutant = x1 + self.F * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.population[i])

                # Selection
                trial_fitness = func(trial)
                self.current_evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                # Check budget
                if self.current_evaluations >= self.budget:
                    return self.population[np.argmin(self.fitness)]

        # Return the best found solution
        return self.population[np.argmin(self.fitness)]