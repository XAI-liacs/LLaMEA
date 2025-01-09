import numpy as np

class EvoOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.cr = 0.9  # Crossover rate for DE
        self.f = 0.8  # Differential weight for DE
        self.w = 0.5  # Inertia weight for PSO
        self.c1 = 1.5  # Cognitive component for PSO
        self.c2 = 1.5  # Social component for PSO

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        fitness_values = [func(individual) for individual in population]
        evaluations = self.population_size

        while evaluations < self.budget:
            new_population = []
            for i in range(self.population_size):
                # DE mutation
                indexes = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indexes, 3, replace=False)]
                mutant = np.clip(population[i] + self.f * (a - population[i] + b - c), -5.0, 5.0)

                # Crossover
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, population[i])
                
                # PSO update
                velocity = self.w * population[i] + self.c1 * np.random.rand(self.dim) * (trial - population[i]) + \
                           self.c2 * np.random.rand(self.dim) * (np.array(population).mean(axis=0) - population[i])
                new_individual = np.clip(population[i] + velocity, -5.0, 5.0)
                
                new_population.append(new_individual)
                evaluations += 1
                if evaluations >= self.budget:
                    break

            population = np.array(new_population)
            fitness_values = [func(individual) for individual in population]

        best_idx = np.argmin(fitness_values)
        return population[best_idx]