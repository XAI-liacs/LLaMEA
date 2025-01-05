import numpy as np

class AdaptiveQuantumGA_CLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(size=(self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.fitness_evaluations = 0
        self.mutation_rate = 0.1

    def chaotic_map(self, x):
        return 4 * x * (1 - x)

    def crossover(self, parent1, parent2):
        alpha = np.random.rand()
        return alpha * parent1 + (1 - alpha) * parent2

    def mutate(self, individual, bounds):
        mutation_vector = self.mutation_rate * np.random.normal(size=self.dim)
        return np.clip(individual + mutation_vector, bounds[0], bounds[1])

    def select_parents(self, fitness):
        probabilities = 1.0 / (1.0 + fitness)
        probabilities /= probabilities.sum()
        return np.random.choice(self.population_size, 2, p=probabilities)

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        fitness = np.array([func(ind) for ind in self.population])
        self.fitness_evaluations += self.population_size

        while self.fitness_evaluations < self.budget:
            new_population = []
            for _ in range(self.population_size):
                p1_idx, p2_idx = self.select_parents(fitness)
                parent1, parent2 = self.population[p1_idx], self.population[p2_idx]
                offspring = self.crossover(parent1, parent2)

                chaotic_factor = self.chaotic_map(np.random.rand())
                if np.random.rand() < chaotic_factor:
                    offspring = self.mutate(offspring, [lower_bound, upper_bound])

                new_population.append(offspring)

            self.population = np.array(new_population)
            fitness = np.array([func(ind) for ind in self.population])
            self.fitness_evaluations += self.population_size

            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < self.best_fitness:
                self.best_fitness = fitness[min_fitness_idx]
                self.best_solution = self.population[min_fitness_idx].copy()

            # Feedback-driven mutation rate adjustment
            diversity = np.std(self.population, axis=0).mean()
            self.mutation_rate = max(0.01, min(0.5, 1.0 / (1.0 + diversity)))

        return self.best_solution