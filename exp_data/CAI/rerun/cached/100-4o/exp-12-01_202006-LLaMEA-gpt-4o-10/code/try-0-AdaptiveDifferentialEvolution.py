import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(20, self.budget // 10)
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        self.f_values = np.full(self.pop_size, np.inf)
        self.evaluations = 0

    def __call__(self, func):
        F = 0.8  # differential weight
        CR = 0.9  # crossover probability

        # Evaluate initial population
        for i in range(self.pop_size):
            self.f_values[i] = func(self.population[i])
            self.evaluations += 1

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                # Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.evaluations += 1

                if f_trial < self.f_values[i]:
                    self.population[i] = trial
                    self.f_values[i] = f_trial

                if self.evaluations >= self.budget:
                    break

        return self.population[np.argmin(self.f_values)]

# Example usage:
# optimizer = AdaptiveDifferentialEvolution(budget=1000, dim=10)
# best_solution = optimizer(some_function)