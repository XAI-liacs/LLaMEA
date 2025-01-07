import numpy as np

class BioInspiredFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 * dim
        self.alpha = 0.5  # Randomness parameter
        self.beta0 = 1.0  # Base attractiveness
        self.gamma = 1.0  # Absorption coefficient

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = population[best_index]

        while evaluations < self.budget:
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitness[i] > fitness[j]:  # Move firefly i towards j
                        r = np.linalg.norm(population[i] - population[j])
                        beta = self.beta0 * np.exp(-self.gamma * r ** 2)
                        attraction = beta * (population[j] - population[i])
                        random_movement = self.alpha * (np.random.uniform(size=self.dim) - 0.5)
                        population[i] += attraction + random_movement
                        population[i] = np.clip(population[i], lb, ub)

                        # Evaluate new position
                        new_fitness = func(population[i])
                        evaluations += 1

                        # Update fitness
                        if new_fitness < fitness[i]:
                            fitness[i] = new_fitness

                        # Update best position
                        if new_fitness < fitness[best_index]:
                            best_index = i
                            best_position = population[i]

                        if evaluations >= self.budget:
                            break

        return best_position, fitness[best_index]