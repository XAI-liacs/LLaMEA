import numpy as np

class SelfAdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 10 * dim
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, dim))
        self.mutation_scale = np.random.uniform(0.5, 1.0, self.pop_size)
        self.crossover_rate = np.random.uniform(0.1, 0.9, self.pop_size)
        self.best_index = None

    def __call__(self, func):
        func_evals = 0
        scores = np.array([func(ind) for ind in self.population])
        func_evals += self.pop_size
        self.f_opt = np.min(scores)
        self.best_index = np.argmin(scores)
        self.x_opt = np.copy(self.population[self.best_index])

        while func_evals < self.budget:
            for i in range(self.pop_size):
                if func_evals >= self.budget:
                    break

                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                while self.best_index in indices:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                
                a, b, c = self.population[indices]
                mutant = a + self.mutation_scale[i] * (b - c)
                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.crossover_rate[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, self.population[i])

                # Selection
                score_trial = func(trial)
                func_evals += 1
                if score_trial < scores[i]:
                    scores[i] = score_trial
                    self.population[i] = trial
                    self.mutation_scale[i] = np.random.uniform(0.5, 1.0)
                    self.crossover_rate[i] = np.random.uniform(0.1, 0.9)
                    if score_trial < self.f_opt:
                        self.f_opt = score_trial
                        self.x_opt = np.copy(trial)
                        self.best_index = i

        return self.f_opt, self.x_opt