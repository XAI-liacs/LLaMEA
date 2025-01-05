import numpy as np

class SelfAdaptiveQIDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_individuals = max(10, min(50, budget // 10))
        self.population = None
        self.fitness = None
        self.best_position = None
        self.best_fitness = float('inf')
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.quantum_prob = 0.1

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.num_individuals, self.dim)

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        for i, f in enumerate(fitness):
            if f < self.best_fitness:
                self.best_fitness = f
                self.best_position = self.population[i]
        return fitness

    def mutation(self, parent_idx):
        indices = [idx for idx in range(self.num_individuals) if idx != parent_idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant_vector = a + self.mutation_factor * (b - c)
        return np.clip(mutant_vector, 0, 1)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        offspring = np.where(crossover_mask, mutant, target)
        return offspring

    def quantum_inspired_mutation(self, lb, ub):
        for i in range(self.num_individuals):
            if np.random.rand() < self.quantum_prob:
                quantum_vector = lb + (ub - lb) * np.random.rand(self.dim)
                self.population[i] = np.mean([self.population[i], quantum_vector], axis=0)
                self.population[i] = np.clip(self.population[i], lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            self.fitness = self.evaluate_population(func)
            evaluations += self.num_individuals

            if evaluations >= self.budget:
                break

            new_population = np.copy(self.population)
            for i in range(self.num_individuals):
                mutant = self.mutation(i)
                offspring = self.crossover(self.population[i], mutant)
                offspring_fitness = func(offspring)
                evaluations += 1

                if offspring_fitness < self.fitness[i]:
                    new_population[i] = offspring
                    self.fitness[i] = offspring_fitness
                    if offspring_fitness < self.best_fitness:
                        self.best_fitness = offspring_fitness
                        self.best_position = offspring

            self.population = new_population
            self.quantum_inspired_mutation(lb, ub)

        return self.best_position, self.best_fitness