import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 5 * dim
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.Inf)
        self.mutation_factor = 0.5
        self.crossover_probability = 0.7
        self.evaluations = 0

    def _evaluate_population(self, func):
        for i in range(self.population_size):
            if self.evaluations < self.budget:
                f = func(self.population[i])
                self.evaluations += 1
                if f < self.fitness[i]:
                    self.fitness[i] = f
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = self.population[i].copy()

    def __call__(self, func):
        self._evaluate_population(func)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                a, b, c = np.random.choice(list(set(range(self.population_size)) - {i}), 3, replace=False)
                mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, -5.0, 5.0)

                trial = np.copy(self.population[i])
                cross_points = np.random.rand(self.dim) < self.crossover_probability
                trial[cross_points] = mutant[cross_points]

                if self.evaluations < self.budget:
                    f_trial = func(trial)
                    self.evaluations += 1
                    if f_trial < self.fitness[i]:
                        self.population[i] = trial
                        self.fitness[i] = f_trial
                        if f_trial < self.f_opt:
                            self.f_opt = f_trial
                            self.x_opt = trial

            # Adaptive parameter adjustment
            self.mutation_factor = np.clip(self.mutation_factor + np.random.normal(0, 0.1), 0.1, 0.9)
            self.crossover_probability = np.clip(self.crossover_probability + np.random.normal(0, 0.1), 0.1, 0.9)

        return self.f_opt, self.x_opt