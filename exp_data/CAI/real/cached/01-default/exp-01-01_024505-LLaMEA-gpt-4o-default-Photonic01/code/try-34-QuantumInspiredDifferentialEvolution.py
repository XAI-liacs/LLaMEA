import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(10, min(50, budget // 10))
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.levy_alpha = 1.5  # Levy flight parameter
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)

    def levy_flight(self):
        u = np.random.normal(0, 1, self.dim) * (1 / np.abs(np.random.normal(0, 1)) ** (1 / self.levy_alpha))
        return u

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        best_index = np.argmin(fitness)
        if fitness[best_index] < self.best_fitness:
            self.best_fitness = fitness[best_index]
            self.best_solution = self.population[best_index]
        self.evaluations += len(fitness)
        return fitness

    def mutate_and_crossover(self, lb, ub):
        new_population = np.empty_like(self.population)
        for i in range(self.pop_size):
            indices = np.random.choice(self.pop_size, 3, replace=False)
            x_r1, x_r2, x_r3 = self.population[indices]
            mutant = x_r1 + self.F * (x_r2 - x_r3) + self.levy_flight()
            mutant = np.clip(mutant, lb, ub)
            trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.population[i])
            new_population[i] = trial
        return new_population

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            fitness = self.evaluate_population(func)
            if self.evaluations >= self.budget:
                break

            offspring = self.mutate_and_crossover(lb, ub)
            offspring_fitness = np.array([func(ind) for ind in offspring])
            self.evaluations += len(offspring_fitness)

            for i in range(self.pop_size):
                if offspring_fitness[i] < fitness[i]:
                    self.population[i] = offspring[i]
                    fitness[i] = offspring_fitness[i]

            best_index = np.argmin(fitness)
            if fitness[best_index] < self.best_fitness:
                self.best_fitness = fitness[best_index]
                self.best_solution = self.population[best_index]

        return self.best_solution, self.best_fitness