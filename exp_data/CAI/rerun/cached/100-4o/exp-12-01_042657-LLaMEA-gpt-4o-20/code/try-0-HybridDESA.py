import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.bounds = (-5.0, 5.0)
        self.f = 0.8  # Differential weight
        self.cr = 0.9  # Crossover probability
        self.temperature = 100.0  # Initial temperature for annealing

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_spent = self.population_size

        while budget_spent < self.budget:
            for i in range(self.population_size):
                if budget_spent >= self.budget:
                    break

                # Differential Evolution mutation and crossover
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.f * (population[b] - population[c])
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
                crossover = np.random.rand(self.dim) < self.cr
                offspring = np.where(crossover, mutant, population[i])

                # Evaluate offspring and apply Simulated Annealing acceptance
                offspring_fitness = func(offspring)
                budget_spent += 1
                delta = offspring_fitness - fitness[i]
                if delta < 0 or np.exp(-delta / self.temperature) > np.random.rand():
                    population[i] = offspring
                    fitness[i] = offspring_fitness

            # Decrease temperature
            self.temperature *= 0.99

        return population[np.argmin(fitness)]

# Example usage:
# optimizer = HybridDESA(budget=10000, dim=10)
# best_solution = optimizer(some_black_box_function)