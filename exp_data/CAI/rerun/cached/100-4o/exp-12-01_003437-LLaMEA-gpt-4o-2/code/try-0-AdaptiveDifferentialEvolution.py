import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        self.best_solution = None
        self.best_score = float('inf')

    def __call__(self, func):
        evals = 0
        scores = np.array([func(ind) for ind in self.population])
        evals += self.population_size
        self.best_score = np.min(scores)
        self.best_solution = self.population[np.argmin(scores)]

        while evals < self.budget:
            for i in range(self.population_size):
                if evals >= self.budget:
                    break
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.bounds[0], self.bounds[1])
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])
                trial_score = func(trial)
                evals += 1
                if trial_score < scores[i]:
                    self.population[i] = trial
                    scores[i] = trial_score
                    if trial_score < self.best_score:
                        self.best_score = trial_score
                        self.best_solution = trial

        return self.best_solution