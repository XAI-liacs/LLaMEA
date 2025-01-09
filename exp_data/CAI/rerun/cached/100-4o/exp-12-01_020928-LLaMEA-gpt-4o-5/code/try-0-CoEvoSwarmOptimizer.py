import numpy as np

class CoEvoSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, dim))
        self.velocity = np.zeros((self.pop_size, dim))
        self.personal_best = np.copy(self.population)
        self.personal_best_value = np.full(self.pop_size, np.inf)
        self.global_best = None
        self.global_best_value = np.inf
        self.f_evals = 0
        self.f_weight = 0.8
        self.c_cognitive = 2.0
        self.c_social = 2.0
        self.c_mutation = 0.9

    def __call__(self, func):
        while self.f_evals < self.budget:
            # Evaluate population
            for i in range(self.pop_size):
                if self.f_evals >= self.budget:
                    break
                value = func(self.population[i])
                self.f_evals += 1

                # Update personal best
                if value < self.personal_best_value[i]:
                    self.personal_best_value[i] = value
                    self.personal_best[i] = self.population[i]

                # Update global best
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best = self.population[i]

            # Evolutionary strategy: Differential Mutation
            indices = np.arange(self.pop_size)
            for i in range(self.pop_size):
                if self.f_evals >= self.budget:
                    break
                a, b, c = np.random.choice(indices[indices != i], 3, replace=False)
                mutant = np.clip(self.population[a] + self.f_weight * (self.population[b] - self.population[c]), -5.0, 5.0)
                cross_points = np.random.rand(self.dim) < self.c_mutation
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])
                value = func(trial)
                self.f_evals += 1

                if value < self.personal_best_value[i]:
                    self.population[i] = trial
                    self.personal_best_value[i] = value
                    self.personal_best[i] = trial
                    if value < self.global_best_value:
                        self.global_best_value = value
                        self.global_best = trial

            # Swarm strategy: Update velocity and position
            r1, r2 = np.random.rand(2, self.pop_size, self.dim)
            self.velocity = (self.velocity
                             + self.c_cognitive * r1 * (self.personal_best - self.population)
                             + self.c_social * r2 * (self.global_best - self.population))
            self.population = np.clip(self.population + self.velocity, -5.0, 5.0)

        return self.global_best

# Example usage:
# optimizer = CoEvoSwarmOptimizer(budget=1000, dim=10)
# best_solution = optimizer(some_black_box_function)