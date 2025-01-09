import numpy as np

class AdaptiveGradientProjectionWithChaoticVariation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(10 * dim, 50)
        self.learning_rate = 0.01
        self.current_evaluations = 0

    def chaotic_map(self, x):
        return 4.0 * x * (1.0 - x)

    def initialize_population(self, bounds):
        lower, upper = bounds.lb, bounds.ub
        return np.random.uniform(lower, upper, (self.population_size, self.dim))

    def evaluate(self, func, population):
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += len(population)
        return fitness

    def gradient_step(self, individual, fitness, bounds):
        perturbation = np.random.normal(0, 0.1, self.dim)
        grad_individual = individual + perturbation
        grad_individual = np.clip(grad_individual, bounds.lb, bounds.ub)
        grad_fitness = func(grad_individual)
        self.current_evaluations += 1

        gradient = (grad_fitness - fitness) / (perturbation + 1e-8)
        step = self.learning_rate * gradient
        new_individual = individual - step
        new_individual = np.clip(new_individual, bounds.lb, bounds.ub)
        return new_individual

    def optimize(self, func, bounds):
        chaotic_sequence = np.random.rand(self.population_size)
        population = self.initialize_population(bounds)
        fitness = self.evaluate(func, population)

        while self.current_evaluations < self.budget:
            new_population = []
            new_fitness = []

            for i in range(self.population_size):
                individual = population[i]
                fitness_val = fitness[i]

                new_individual = self.gradient_step(individual, fitness_val, bounds)

                chaotic_sequence[i] = self.chaotic_map(chaotic_sequence[i])
                chaotic_perturbation = chaotic_sequence[i] * (bounds.ub - bounds.lb) * np.random.uniform(0.005, 0.015)
                new_individual += chaotic_perturbation
                new_individual = np.clip(new_individual, bounds.lb, bounds.ub)

                new_fitness_val = func(new_individual)
                self.current_evaluations += 1

                if new_fitness_val < fitness_val:
                    new_population.append(new_individual)
                    new_fitness.append(new_fitness_val)
                else:
                    new_population.append(individual)
                    new_fitness.append(fitness_val)

                if self.current_evaluations >= self.budget:
                    break

            population = np.array(new_population)
            fitness = np.array(new_fitness)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def __call__(self, func):
        bounds = func.bounds
        best_solution, best_value = self.optimize(func, bounds)
        return best_solution, best_value