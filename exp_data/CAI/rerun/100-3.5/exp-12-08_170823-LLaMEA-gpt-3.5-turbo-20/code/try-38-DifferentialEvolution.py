import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.CR = 0.9  # Crossover rate
        self.F = 0.5   # Differential weight
        self.pop_size = self.budget
        self.crowding_factor = 0.5

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])

        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)

                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[j])

                f_trial = func(trial)
                if f_trial < fitness[j]:
                    population[j] = trial
                    fitness[j] = f_trial

                if fitness[j] < self.f_opt:
                    self.f_opt = fitness[j]
                    self.x_opt = population[j]

            # Preserve population diversity using crowding distance
            distances = np.linalg.norm(population - np.mean(population, axis=0), axis=1)
            sorted_indices = np.argsort(distances)
            for k in range(self.pop_size):
                if k == 0 or k == self.pop_size - 1:
                    continue
                crowding_factor = (distances[sorted_indices[k + 1]] - distances[sorted_indices[k - 1]]) / (distances[sorted_indices[-1]] - distances[sorted_indices[0]])
                if np.random.rand() < crowding_factor * self.crowding_factor:
                    population[sorted_indices[k]] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)

        return self.f_opt, self.x_opt