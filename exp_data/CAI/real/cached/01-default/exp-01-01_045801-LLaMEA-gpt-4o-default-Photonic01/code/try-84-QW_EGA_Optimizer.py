import numpy as np

class QW_EGA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate_min = 0.01
        self.mutation_rate_max = 0.2
        self.crossover_rate_min = 0.6
        self.crossover_rate_max = 0.9
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness = np.array([self.evaluate(ind) for ind in self.population])
        self.update_best()

    def evaluate(self, solution):
        return self.func(solution)

    def select_parents(self):
        fitness_inv = 1 / (self.fitness + 1e-12)
        probabilities = fitness_inv / fitness_inv.sum()
        parents_idx = np.random.choice(range(self.population_size), size=2, replace=False, p=probabilities)
        return self.population[parents_idx]

    def crossover(self, parent1, parent2):
        crossover_rate = np.random.uniform(self.crossover_rate_min, self.crossover_rate_max)
        mask = np.random.rand(self.dim) < crossover_rate
        offspring = np.where(mask, parent1, parent2)
        return offspring

    def mutate(self, individual):
        mutation_rate = np.random.uniform(self.mutation_rate_min, self.mutation_rate_max)
        mutation_mask = np.random.rand(self.dim) < mutation_rate
        noise = np.random.normal(0, 1, self.dim)
        individual = individual + mutation_mask * noise
        return individual

    def quantum_wave_adjustment(self):
        wave_patterns = np.sin(np.linspace(0, np.pi, self.population_size))
        self.mutation_rate_min = 0.01 + 0.1 * wave_patterns.min()
        self.mutation_rate_max = 0.2 + 0.1 * wave_patterns.max()
        self.crossover_rate_min = 0.6 + 0.1 * wave_patterns.min()
        self.crossover_rate_max = 0.9 + 0.1 * wave_patterns.max()

    def update_best(self):
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_fitness:
            self.best_solution = self.population[best_idx].copy()
            self.best_fitness = self.fitness[best_idx]

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = self.select_parents()
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutate(offspring)
                offspring = np.clip(offspring, lb, ub)
                new_population.append(offspring)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

            self.population = np.array(new_population)
            self.fitness = np.array([self.evaluate(ind) for ind in self.population])
            self.update_best()
            self.quantum_wave_adjustment()

        return {'solution': self.best_solution, 'fitness': self.best_fitness}