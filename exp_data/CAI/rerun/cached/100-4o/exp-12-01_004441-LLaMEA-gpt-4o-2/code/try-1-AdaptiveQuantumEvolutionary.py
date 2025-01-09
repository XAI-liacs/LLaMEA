import numpy as np

class AdaptiveQuantumEvolutionary:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = float('inf')

    def __call__(self, func):
        evaluations = 0

        def quantum_rotation_gate(x, best, worst, iter_max, iter_curr):
            alpha = np.pi * (1 - iter_curr / iter_max)
            new_x = x + np.sin(alpha) * (best - worst)
            return np.clip(new_x, self.lower_bound, self.upper_bound)

        max_iterations = self.budget // self.population_size
        F = 0.5  # Differential evolution scaling factor
        CR = 0.7  # Crossover probability

        for iteration in range(max_iterations):
            fitness = np.array([func(ind) for ind in self.population])
            evaluations += self.population_size

            if evaluations >= self.budget:
                break

            best_idx = np.argmin(fitness)
            worst_idx = np.argmax(fitness)

            if fitness[best_idx] < self.best_fitness:
                self.best_fitness = fitness[best_idx]
                self.best_solution = self.population[best_idx].copy()

            new_population = []

            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]

                mutant_vector = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < CR, mutant_vector, self.population[i])

                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    new_population.append(trial_vector)
                else:
                    new_population.append(self.population[i])

                if evaluations >= self.budget:
                    break

            self.population = np.array(new_population)

            # Apply quantum rotation gate for better exploration
            self.population = np.array([
                quantum_rotation_gate(ind, self.population[best_idx], self.population[worst_idx], max_iterations, iteration)
                for ind in self.population
            ])

        return self.best_solution