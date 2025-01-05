import numpy as np

class DESAOptimizer:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evals = self.pop_size

        while num_evals < self.budget:
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + 0.8 * (b - c), bounds[0], bounds[1])
                
                cross_points = np.random.rand(self.dim) < 0.9
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                trial_fitness = func(trial)
                num_evals += 1

                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / (0.1 * (self.budget - num_evals) / self.budget)):
                    population[i] = trial
                    fitness[i] = trial_fitness

                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial

                if num_evals >= self.budget:
                    break

        return self.f_opt, self.x_opt