import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(5 * dim, 50)
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = np.random.uniform(low=self.lower_bound, high=self.upper_bound,
                                            size=(self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf

    def __call__(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        evaluations = self.population_size
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.population[indices]
                mutant = np.clip(x0 + self.F * (x1 - x2), self.lower_bound, self.upper_bound)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, self.population[i])
                
                # Local search (adaptive)
                if np.random.rand() < 0.2:
                    jitter = np.random.normal(0, 0.1, self.dim)
                    trial = np.clip(trial + jitter, self.lower_bound, self.upper_bound)
                
                trial_fitness = func(trial)
                evaluations += 1
                
                # Selection
                if trial_fitness < fitness[i]:
                    self.population[i] = trial
                    fitness[i] = trial_fitness
                
                # Update global best
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial

        return self.best_solution, self.best_fitness