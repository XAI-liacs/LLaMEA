import numpy as np

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 10 * dim
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.fitness = np.inf * np.ones(self.population_size)
        self.best_solution = None
        self.best_fitness = np.inf
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            self.population = self.opposition_based_learning(self.population)
            for i in range(self.population_size):
                trial_vector = self.mutate_and_crossover(i)
                trial_vector = np.clip(trial_vector, self.bounds[0], self.bounds[1])
                trial_fitness = func(trial_vector)
                self.evaluations += 1
                
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness
                    
                    if trial_fitness < self.best_fitness:
                        self.best_solution = trial_vector
                        self.best_fitness = trial_fitness

                if self.evaluations >= self.budget:
                    break

        return self.best_solution

    def opposition_based_learning(self, population):
        new_population = population.copy()
        for i in range(self.population_size):
            opposite = self.bounds[0] + self.bounds[1] - population[i]
            if np.random.rand() < 0.5:
                new_population[i] = opposite
        return new_population

    def mutate_and_crossover(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        F = 0.8 if self.evaluations % 2 == 0 else 0.5  # Adaptive differential factor
        mutant_vector = self.population[a] + F * (self.population[b] - self.population[c])
        crossover_rate = 0.9
        trial_vector = np.where(np.random.rand(self.dim) < crossover_rate, mutant_vector, self.population[idx])
        return trial_vector