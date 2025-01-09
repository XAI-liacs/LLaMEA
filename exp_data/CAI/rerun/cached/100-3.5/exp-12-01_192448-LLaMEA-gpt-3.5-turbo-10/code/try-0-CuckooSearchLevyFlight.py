import numpy as np

class CuckooSearchLevyFlight:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def levy_flight(size):
            sigma = (np.math.gamma(1 + 1.5) * np.math.sin(np.pi * 1.5 / 2) / (
                np.math.gamma((1 + 1.5) / 2) * 1.5 * 2 ** ((1.5 - 1) / 2))) ** (1 / 1.5)
            u = np.random.normal(0, sigma, size)
            v = np.random.normal(0, 1, size)
            step = u / (np.abs(v) ** (1 / 1.5))
            return step

        def optimize():
            population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
            fitness_values = [func(individual) for individual in population]

            for _ in range(self.budget):
                index = np.argmax(fitness_values)
                cuckoo = population[index] + levy_flight(self.dim)
                cuckoo_fitness = func(cuckoo)

                if cuckoo_fitness > fitness_values[index]:
                    population[index] = cuckoo
                    fitness_values[index] = cuckoo_fitness

            best_index = np.argmax(fitness_values)
            return population[best_index], fitness_values[best_index]

        best_solution, best_fitness = optimize()
        return best_solution, best_fitness