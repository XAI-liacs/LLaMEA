import numpy as np

class AdaptiveOppositionDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 10 * dim
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.best_solution = None
        self.best_fitness = np.inf
        self.CR = 0.9  # Crossover probability
        self.F = 0.8   # Differential weight
        self.evaluations = 0

    def opposition_based_learning(self):
        opposite_population = self.bounds[0] + self.bounds[1] - self.population
        return opposite_population

    def evaluate(self, func):
        for i in range(self.population_size):
            if self.evaluations < self.budget:
                if self.fitness[i] == np.inf:
                    self.fitness[i] = func(self.population[i])
                    self.evaluations += 1
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_solution = self.population[i].copy()

    def differential_evolution(self, func):
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                a, b, c = self.population[np.random.choice(self.population_size, 3, replace=False)]
                mutant_vector = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])
                
                trial_vector = np.where(np.random.rand(self.dim) < self.CR, mutant_vector, self.population[i])
                trial_fitness = func(trial_vector)
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial_vector.copy()

    def __call__(self, func):
        self.evaluate(func)
        opposite_population = self.opposition_based_learning()
        for i in range(self.population_size):
            if self.evaluations < self.budget:
                opposite_fitness = func(opposite_population[i])
                self.evaluations += 1
                if opposite_fitness < self.fitness[i]:
                    self.population[i] = opposite_population[i]
                    self.fitness[i] = opposite_fitness
                    if opposite_fitness < self.best_fitness:
                        self.best_fitness = opposite_fitness
                        self.best_solution = opposite_population[i].copy()
        
        self.differential_evolution(func)
        return self.best_solution, self.best_fitness