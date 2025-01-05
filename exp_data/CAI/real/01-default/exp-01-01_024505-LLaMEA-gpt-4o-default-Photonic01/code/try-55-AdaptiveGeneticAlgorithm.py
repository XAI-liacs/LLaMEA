import numpy as np

class AdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, min(50, budget // 10))
        self.population = None
        self.fitness = None
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elitism_count = max(1, self.population_size // 10)

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def evaluate_population(self, func):
        self.fitness = np.array([func(individual) for individual in self.population])

    def select_parents(self):
        selection_prob = 1.0 / (1.0 + self.fitness)
        selection_prob /= selection_prob.sum()
        parent_indices = np.random.choice(np.arange(self.population_size), size=self.population_size, p=selection_prob)
        return self.population[parent_indices]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.dim - 1)
            return np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return parent1 if np.random.rand() > 0.5 else parent2

    def mutate(self, individual, lb, ub):
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                individual[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()
        return np.clip(individual, lb, ub)

    def generate_offspring(self, parents, lb, ub):
        offspring = []
        for i in range(0, self.population_size, 2):
            parent1, parent2 = parents[i], parents[min(i + 1, self.population_size - 1)]
            child1 = self.crossover(parent1, parent2)
            child2 = self.crossover(parent2, parent1)
            offspring.append(self.mutate(child1, lb, ub))
            offspring.append(self.mutate(child2, lb, ub))
        return np.array(offspring)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += self.population_size

            if evaluations >= self.budget:
                break

            best_indices = np.argsort(self.fitness)[:self.elitism_count]
            elite_individuals = self.population[best_indices]
            parents = self.select_parents()
            offspring = self.generate_offspring(parents, lb, ub)
            self.population = np.concatenate((elite_individuals, offspring[:self.population_size - self.elitism_count]))

            self.mutation_rate = 0.1 - 0.09 * (evaluations / self.budget)

        best_index = np.argmin(self.fitness)
        return self.population[best_index], self.fitness[best_index]