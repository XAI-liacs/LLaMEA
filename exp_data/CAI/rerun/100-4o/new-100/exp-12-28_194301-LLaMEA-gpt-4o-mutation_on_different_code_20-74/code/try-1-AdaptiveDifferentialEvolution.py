import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 20
        self.pop_size = self.initial_pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.mutation_rate = 0.5
        self.crossover_rate = 0.9
    
    def _mutate(self, population, best_idx):
        idxs = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = population[idxs]
        best = population[best_idx]
        return best + self.mutation_rate * (a - b + c - best)
    
    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return np.clip(trial, -5.0, 5.0)
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        evals = self.pop_size

        while evals < self.budget:
            for i in range(self.pop_size):
                mutant = self._mutate(population, best_idx)
                trial = self._crossover(population[i], mutant)
                f_trial = func(trial)

                evals += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        best_idx = i
                if evals >= self.budget:
                    break

            self.mutation_rate = max(0.1, self.mutation_rate * 0.9 + 0.1 * (np.random.rand() * 0.5))
            self.crossover_rate = max(0.4, self.crossover_rate * 0.8 + 0.2 * (np.random.rand() * 0.2))

            # Dynamic population adjustment
            if evals % (self.budget // 10) == 0:
                self.pop_size = max(5, int(self.initial_pop_size * (1 - evals / self.budget)))
                elite = population[fitness.argsort()[:self.pop_size]]
                population = np.vstack((elite, np.random.uniform(-5.0, 5.0, (self.initial_pop_size - self.pop_size, self.dim))))
                fitness = np.array([func(ind) for ind in population])
                best_idx = np.argmin(fitness)

        return self.f_opt, self.x_opt