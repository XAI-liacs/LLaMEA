import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4 * self.dim, 20)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0
        self.dynamic_shrinkage_rate = 0.92  # Slightly faster shrinkage
        self.elitism_rate = 0.07  # Increased elitism rate for robust solutions
        self.learning_rate = 0.1  # New parameter for adaptive mutation and crossover

    def __call__(self, func):
        best_solution = None
        best_fitness = np.inf

        # Initial fitness evaluation
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1
            if self.fitness[i] < best_fitness:
                best_fitness = self.fitness[i]
                best_solution = self.population[i]

        F_memory = np.full(self.population_size, 0.5)
        CR_memory = np.full(self.population_size, 0.9)

        while self.evaluations < self.budget:
            # Dynamically resize population
            if self.evaluations % (self.budget // (self.population_size // 5)) == 0:
                self.population_size = int(self.population_size * self.dynamic_shrinkage_rate)
                self.fitness = self.fitness[:self.population_size]
                self.population = self.population[:self.population_size]

            elitism_count = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(self.fitness)[:elitism_count]
            elite_population = self.population[elite_indices]

            for i in range(self.population_size):
                F = F_memory[i] + self.learning_rate * np.random.uniform(-0.1, 0.1)
                CR = CR_memory[i] + self.learning_rate * np.random.uniform(-0.1, 0.1)
                F = np.clip(F, 0.4, 0.9)
                CR = np.clip(CR, 0.7, 1.0)

                indices = np.random.choice([x for x in range(self.population_size) if x != i], 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)

                trial = np.copy(self.population[i])
                crossover_points = np.random.rand(self.dim) < CR
                if not np.any(crossover_points):
                    crossover_points[np.random.randint(self.dim)] = True
                trial[crossover_points] = mutant[crossover_points]

                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial
                    F_memory[i] = F
                    CR_memory[i] = CR
                if self.evaluations >= self.budget:
                    break

            # Integrating elitism: replace part of the population with best solutions
            self.population[:elitism_count] = elite_population

        return best_solution