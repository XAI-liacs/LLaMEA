import numpy as np

class ChaoticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.crossover_rate = 0.85  # Adjusted dynamically in the loop
        self.F = 0.5
        self.lb, self.ub = None, None
        self.chaotic_seed = 0.5678

    def chaotic_sequence(self, seed, n):
        x = seed
        result = []
        for _ in range(n):
            x = 3.91 * x * (1.0 - x)
            result.append(x)
        return result

    def initialize_population(self):
        chaotic_seq = self.chaotic_sequence(self.chaotic_seed, self.population_size * self.dim)
        population = np.array(chaotic_seq).reshape(self.population_size, self.dim)
        return self.lb + population * (self.ub - self.lb)

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population()
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        evaluations = self.population_size
        adaptive_F = self.F

        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                d = np.random.choice(indices)
                adaptive_F = 0.4 + 0.6 * np.random.rand()  # Adaptive mutation factor
                mutant = population[a] + adaptive_F * (population[b] - population[c]) + 0.1 * (population[d] - population[i])
                mutant = np.clip(mutant, self.lb, self.ub)

                self.crossover_rate = 0.7 + 0.3 * (fitness[i] - fitness[best_idx]) / (fitness[i] + fitness[best_idx])
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_solution = trial
                        adaptive_F *= 0.95

                if evaluations >= self.budget:
                    break

        return best_solution