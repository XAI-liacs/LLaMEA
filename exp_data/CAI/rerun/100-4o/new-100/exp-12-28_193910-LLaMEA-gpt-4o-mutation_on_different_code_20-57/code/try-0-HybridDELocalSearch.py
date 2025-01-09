import numpy as np

class HybridDELocalSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = min(50, self.budget)  # Size of the population
        self.bounds = [-5.0, 5.0]
        self.F = 0.8
        self.CR = 0.9

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.full(self.population_size, np.inf)

        # Evaluate initial population
        for i in range(self.population_size):
            fitness[i] = func(population[i])
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = population[i]

        evaluations = self.population_size

        # Begin optimization loop
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation and Crossover
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + self.F * (x2 - x3), self.bounds[0], self.bounds[1])
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])

                # Evaluate trial vector
                f_trial = func(trial)
                evaluations += 1

                # Selection
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial

                    # Local Search (Simplex or Gradient Descent can be incorporated here)
                    trial_local = self.local_search(trial, func)
                    f_trial_local = func(trial_local)
                    evaluations += 1

                    if f_trial_local < f_trial:
                        population[i] = trial_local
                        fitness[i] = f_trial_local

                # Update best found so far
                if fitness[i] < self.f_opt:
                    self.f_opt = fitness[i]
                    self.x_opt = population[i]

                # Break if budget is exhausted
                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt

    def local_search(self, x, func):
        # Simple local search: gradient-free small perturbations
        perturbation = np.random.uniform(-0.1, 0.1, self.dim)
        x_new = np.clip(x + perturbation, self.bounds[0], self.bounds[1])
        return x_new