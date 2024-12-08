import numpy as np

class AdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.current_evals = 0

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def evaluate_population(self, population, func):
        return np.apply_along_axis(func, 1, population)

    def select_parents(self, population, fitness):
        indices = np.argsort(fitness)[:self.population_size // 3]  # Select top third
        return population[indices]

    def crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dim)
        return (1 - alpha) * parent1 + alpha * parent2

    def mutate(self, individual):
        mutation_vector = np.random.uniform(-1, 1, self.dim) * self.mutation_rate
        return np.clip(individual + mutation_vector, self.lower_bound, self.upper_bound)

    def run_generation(self, population, func):
        fitness = self.evaluate_population(population, func)
        parents = self.select_parents(population, fitness)
        new_population = []

        for _ in range(self.population_size // 2):  # Reduce loop count for efficiency
            indices = np.random.choice(len(parents), 2, replace=False)
            parent1, parent2 = parents[indices]
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
            self.current_evals += 1
            if self.current_evals >= self.budget:
                break

        return np.array(new_population)

    def update_parameters(self, population):
        diversity = np.mean(np.std(population, axis=0))
        self.mutation_rate = max(0.01, 1 / (5 * self.dim) * diversity)
        self.population_size = int(max(10, min(100, self.population_size * (1 + 0.05 * (diversity - 0.1)))))

    def __call__(self, func):
        population = self.initialize_population()

        while self.current_evals < self.budget:
            population = self.run_generation(population, func)
            self.update_parameters(population)

        fitness = self.evaluate_population(population, func)
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]