import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 10  # Initial population size
        self.cr = 0.5  # Crossover probability
        self.f_scale = 0.5  # Scaling factor

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))

        for i in range(self.budget):
            for j in range(self.pop_size):
                idxs = list(range(self.pop_size))
                idxs.remove(j)
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = population[a] + self.f_scale * (population[b] - population[c])

                crossover_points = np.random.rand(self.dim) < self.cr
                if not np.any(crossover_points):
                    crossover_points[np.random.randint(0, self.dim)] = True

                trial = np.where(crossover_points, mutant, population[j])
                f_trial = func(trial)

                if f_trial < func(population[j]):
                    population[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

        return self.f_opt, self.x_opt