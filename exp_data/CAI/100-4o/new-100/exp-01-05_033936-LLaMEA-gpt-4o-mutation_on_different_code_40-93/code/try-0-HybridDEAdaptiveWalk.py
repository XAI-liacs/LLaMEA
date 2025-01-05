import numpy as np

class HybridDEAdaptiveWalk:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 20
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.step_size = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        evals = self.pop_size

        while evals < self.budget:
            if evals % (self.pop_size * 10) == 0:
                diversity = np.mean(np.std(population, axis=0))
                self.mutation_factor = max(0.5, min(1.0, diversity * 2))
                self.step_size = max(0.01, diversity / 10)

            new_population = np.copy(population)

            for i in range(self.pop_size):
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = a + self.mutation_factor * (b - c)
                mutant = np.clip(mutant, lb, ub)

                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                if np.random.rand() < 0.1:
                    trial += np.random.normal(0, self.step_size, self.dim)
                    trial = np.clip(trial, lb, ub)

                f = func(trial)
                evals += 1

                if f < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = f
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial

                if evals >= self.budget:
                    break

            population = new_population

        return self.f_opt, self.x_opt