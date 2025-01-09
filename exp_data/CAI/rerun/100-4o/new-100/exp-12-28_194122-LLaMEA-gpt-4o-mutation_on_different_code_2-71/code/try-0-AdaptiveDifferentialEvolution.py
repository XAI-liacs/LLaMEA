import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def mutate(self, population, best_idx):
        a, b, c = np.random.choice(population.shape[0], 3, replace=False)
        x_best = population[best_idx]
        return x_best + self.F * (population[a] - population[b] + population[c] - x_best)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        eval_count = self.pop_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        while eval_count < self.budget:
            for i in range(self.pop_size):
                mutant = self.mutate(population, best_idx)
                mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])
                trial = self.crossover(population[i], mutant)
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

                if eval_count >= self.budget:
                    break

            best_idx = np.argmin(fitness)

        return self.f_opt, self.x_opt