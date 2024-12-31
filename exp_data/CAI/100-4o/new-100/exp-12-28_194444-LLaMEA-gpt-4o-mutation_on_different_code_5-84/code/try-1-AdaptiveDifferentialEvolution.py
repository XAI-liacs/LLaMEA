import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        evals = self.population_size
        diff_weight_initial = 0.8
        crossover_rate_initial = 0.9

        while evals < self.budget:
            diversity = np.mean(np.std(population, axis=0))
            diff_weight = diff_weight_initial * (1 - diversity / self.dim)
            crossover_rate = crossover_rate_initial * (1 - diversity / self.dim)

            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c, d = population[np.random.choice(indices, 4, replace=False)]
                mutant = np.clip(a + diff_weight * (b - c) + 0.5 * (d - a), bounds[0], bounds[1])
                cross_points = np.random.rand(self.dim) < crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                f_trial = func(trial)
                evals += 1

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if evals >= self.budget:
                    break

        return self.f_opt, self.x_opt