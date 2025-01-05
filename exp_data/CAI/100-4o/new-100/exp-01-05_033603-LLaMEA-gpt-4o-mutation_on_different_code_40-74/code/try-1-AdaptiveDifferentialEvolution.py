import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= self.population_size

        # Set initial parameters
        F = 0.5
        CR = 0.5

        while self.budget > 0:
            success_count = 0
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)

                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                self.budget -= 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    success_count += 1

                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

            if success_count:
                F = min(0.9, F + 0.1 * success_count/self.population_size)
                CR = min(0.9, CR + 0.1 * success_count/self.population_size)
            else:
                F = max(0.1, F - 0.1)
                CR = max(0.1, CR - 0.1)

            # Greedy local search
            for j in range(self.population_size):
                local_search = np.clip(population[j] + np.random.normal(0, 0.1, self.dim), func.bounds.lb, func.bounds.ub)
                local_fitness = func(local_search)
                self.budget -= 1

                if local_fitness < fitness[j]:
                    population[j] = local_search
                    fitness[j] = local_fitness

                    if local_fitness < self.f_opt:
                        self.f_opt = local_fitness
                        self.x_opt = local_search

        return self.f_opt, self.x_opt