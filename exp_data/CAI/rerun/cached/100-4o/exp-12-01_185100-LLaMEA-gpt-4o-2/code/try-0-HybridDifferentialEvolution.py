import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim, population_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = population_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.evaluations = 0
        self.best_solution = None
        self.best_fitness = np.inf

    def levy_flight(self, scale=0.1):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v) ** (1 / beta)
        return scale * step

    def mutate(self, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant_vector = self.population[a] + self.F * (self.population[b] - self.population[c])
        mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
        return mutant_vector

    def crossover(self, target_vector, mutant_vector):
        crossover_vector = np.copy(target_vector)
        crossover_points = np.random.rand(self.dim) < self.CR
        if not np.any(crossover_points):
            crossover_points[np.random.randint(0, self.dim)] = True
        crossover_vector[crossover_points] = mutant_vector[crossover_points]
        return crossover_vector

    def __call__(self, func):
        while self.evaluations < self.budget:
            for idx in range(self.population_size):
                target_vector = self.population[idx]
                mutant_vector = self.mutate(idx)
                crossover_vector = self.crossover(target_vector, mutant_vector)

                if np.random.rand() < 0.5:  # Adaptive exploration/exploitation toggle
                    crossover_vector += self.levy_flight()

                trial_fitness = func(crossover_vector)
                self.evaluations += 1

                if trial_fitness < func(target_vector):
                    self.population[idx] = crossover_vector
                    if trial_fitness < self.best_fitness:
                        self.best_solution = crossover_vector
                        self.best_fitness = trial_fitness
            
            if self.evaluations >= self.budget:
                break

        return self.best_solution, self.best_fitness