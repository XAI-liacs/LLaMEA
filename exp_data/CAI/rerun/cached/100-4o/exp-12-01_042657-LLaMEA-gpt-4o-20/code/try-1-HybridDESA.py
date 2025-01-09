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
    
    def adapt_differential_weight(self, iteration, max_iterations):
        return self.f * (1 - iteration / max_iterations)
    
    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_spent = self.population_size
        iteration = 0
        max_iterations = self.budget // self.population_size

        while budget_spent < self.budget:
            iteration += 1
            for i in range(self.population_size):
                if budget_spent >= self.budget:
                    break

                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                adaptive_f = self.adapt_differential_weight(iteration, max_iterations)
                mutant = population[a] + adaptive_f * (population[b] - population[c])
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
                crossover = np.random.rand(self.dim) < self.cr
                offspring = np.where(crossover, mutant, population[i])

                offspring_fitness = func(offspring)
                budget_spent += 1
                if offspring_fitness < fitness[i]:
                    population[i] = offspring
                    fitness[i] = offspring_fitness
                delta = offspring_fitness - fitness[i]
                if delta < 0 or np.exp(-delta / self.temperature) > np.random.rand():
                    population[i] = offspring
                    fitness[i] = offspring_fitness

            self.temperature *= 0.99

        return population[np.argmin(fitness)]

# Example usage:
# optimizer = HybridDESA(budget=10000, dim=10)
# best_solution = optimizer(some_black_box_function)