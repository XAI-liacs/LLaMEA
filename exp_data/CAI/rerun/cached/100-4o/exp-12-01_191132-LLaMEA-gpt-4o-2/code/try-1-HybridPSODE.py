import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 20
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = np.copy(self.population)
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best = None
        self.global_best_value = np.inf
        self.f = 0.8  # Differential evolution scaling factor
        self.cr = 0.9 # Crossover probability
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5 # Cognitive parameter
        self.c2 = 1.5 # Social parameter

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Evaluate current position
                current_value = func(self.population[i])
                evaluations += 1

                # Update personal best
                if current_value < self.personal_best_values[i]:
                    self.personal_best_values[i] = current_value
                    self.personal_best[i] = self.population[i].copy()

                # Update global best
                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best = self.population[i].copy()
                
                if evaluations >= self.budget:
                    break

                # Hybrid PSO-DE update
                # Differential Evolution like mutation
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

                # Trial evaluation
                trial_value = func(trial)
                evaluations += 1

                # Select between trial and current
                if trial_value < current_value:
                    self.population[i] = trial
                    current_value = trial_value

                # Update velocities and positions for PSO
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.personal_best[i] - self.population[i]) +
                                      self.c2 * r2 * (self.global_best - self.population[i]))
                self.population[i] = np.clip(self.population[i] + self.velocities[i], self.bounds[0], self.bounds[1])

        return self.global_best, self.global_best_value