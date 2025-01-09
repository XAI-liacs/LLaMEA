import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.cr = 0.9  # Initial crossover rate
        self.f = 0.5   # Initial differential weight

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                self.cr = 0.1 + np.random.rand() * 0.8  # Adaptive crossover rate
                self.f = 0.4 + np.random.rand() * 0.6   # Adaptive differential weight

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                mutant = np.clip(a + self.f * (b - c), func.bounds.lb, func.bounds.ub)

                crossover_mask = np.random.rand(self.dim) < self.cr
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover_mask, mutant, population[i])

                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial

                if eval_count >= self.budget:
                    break

        return self.f_opt, self.x_opt