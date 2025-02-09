import numpy as np

class AdaptiveLevyDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.crossover_rate = 0.9
        self.mutation_factor = 0.8
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize population
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += self.population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        # Levy distribution parameters
        beta = 1.5
        alpha = 0.01

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation using Levy flight
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                levy_step = alpha * self.levy_flight(beta, self.dim, lb, ub)
                mutant = np.clip(a + levy_step * (b - c), lb, ub)

                # Fitness-guided crossover
                fitness_ratio = fitness[i] / max(fitness)
                current_crossover_rate = self.crossover_rate * (1 - fitness_ratio)
                cross_points = np.random.rand(self.dim) < current_crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update best solution
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

        return best_solution

    def levy_flight(self, beta, dim, lb, ub):
        # Generate Levy flight step
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=dim)
        v = np.random.normal(0, 1, size=dim)
        step = u / np.abs(v) ** (1 / beta)
        return step