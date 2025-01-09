import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.initial_population_size = 20
        self.population = np.random.uniform(-5, 5, (self.initial_population_size, dim))
        self.scores = np.full(self.initial_population_size, np.Inf)
        self.evaluations = 0

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)

        for i in range(len(self.population)):
            self.scores[i] = func(self.population[i])
            self.evaluations += 1

        while self.evaluations < self.budget:
            for i in range(len(self.population)):
                if self.evaluations >= self.budget:
                    break

                indices = [idx for idx in range(len(self.population)) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                
                mutation_factor = 0.5 + np.random.rand() * 0.5
                mutant_vector = np.clip(a + mutation_factor * (b - c), *bounds)
                crossover_rate = 0.7 + 0.3 * (1 - np.min(self.scores) / self.f_opt) if self.f_opt != np.Inf else 0.7
                trial_vector = np.array([
                    mutant_vector[j] if np.random.rand() < crossover_rate else self.population[i, j]
                    for j in range(self.dim)
                ])
                
                trial_score = func(trial_vector)
                self.evaluations += 1

                if trial_score < self.scores[i]:
                    self.population[i] = trial_vector
                    self.scores[i] = trial_score

                    if trial_score < self.f_opt:
                        self.f_opt, self.x_opt = trial_score, trial_vector

            # Dynamic population adaptation
            progress = np.min(self.scores) / self.f_opt if self.f_opt != np.Inf else 1.0
            self.population = self.population[:int(self.initial_population_size * (0.5 + 0.5 * progress))]
            self.scores = self.scores[:len(self.population)]

        return self.f_opt, self.x_opt