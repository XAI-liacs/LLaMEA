import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 10 * self.dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        # Diversity threshold for reinitialization
        self.diversity_threshold = 0.1

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]

        evals = self.pop_size

        while evals < self.budget:
            for i in range(self.pop_size):
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                x1, x2, x3 = population[idxs]
                mutant = np.clip(x1 + self.mutation_factor * (x2 - x3), bounds[:, 0], bounds[:, 1])

                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

                self.mutation_factor = 0.5 + 0.5 * (1 - evals / self.budget)
                self.crossover_rate = 0.9 * (1 - evals / self.budget) + 0.1

                if np.std(population) < self.diversity_threshold:
                    # Reinitialize a portion of the population to preserve diversity
                    num_reinit = self.pop_size // 5
                    reinit_indices = np.random.choice(self.pop_size, num_reinit, replace=False)
                    population[reinit_indices] = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_reinit, self.dim))
                    fitness[reinit_indices] = np.array([func(ind) for ind in population[reinit_indices]])
                    evals += num_reinit

                if evals >= self.budget:
                    break

        return self.f_opt, self.x_opt