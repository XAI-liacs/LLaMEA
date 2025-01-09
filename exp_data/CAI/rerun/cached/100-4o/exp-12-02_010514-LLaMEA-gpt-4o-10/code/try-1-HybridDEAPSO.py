import numpy as np

class HybridDEAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 + int(2 * np.sqrt(self.dim))
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.zeros((self.pop_size, self.dim))
        self.personal_best = self.population.copy()
        self.global_best = None
        self.global_best_value = np.inf
        self.cr = 0.9  # Crossover probability for DE
        self.f = 0.8   # Scaling factor for DE
        self.w = 0.7   # Inertia weight for PSO
        self.c1 = 1.5  # Cognitive component for PSO
        self.c2 = 1.5  # Social component for PSO

    def __call__(self, func):
        evaluations = 0

        # Initialize personal bests
        personal_best_values = np.array([func(ind) for ind in self.population])
        evaluations += self.pop_size
        self.global_best_value = np.min(personal_best_values)
        self.global_best = self.population[np.argmin(personal_best_values)].copy()

        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Differential Evolution mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = self.population[indices]
                self.f = 0.5 + 0.3 * np.random.rand()  # Dynamic scaling factor
                mutant_vector = np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < (self.cr + 0.1 * np.random.rand())  # Adaptive crossover
                trial_vector = np.where(cross_points, mutant_vector, self.population[i])

                # Evaluate trial vector
                trial_value = func(trial_vector)
                evaluations += 1

                # Selection
                if trial_value < personal_best_values[i]:
                    self.population[i] = trial_vector
                    personal_best_values[i] = trial_value
                    self.personal_best[i] = trial_vector

                    # Update global best
                    if trial_value < self.global_best_value:
                        self.global_best_value = trial_value
                        self.global_best = trial_vector

                # Particle Swarm Optimization update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + self.c1 * r1 * (self.personal_best[i] - self.population[i])
                    + self.c2 * r2 * (self.global_best - self.population[i])
                )
                self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lower_bound, self.upper_bound)

            # Adjust parameters dynamically (optional, could be implemented for adaptive behavior)
            # self.w = max(0.4, self.w * 0.995)  # Example: decay inertia weight

        return self.global_best, self.global_best_value