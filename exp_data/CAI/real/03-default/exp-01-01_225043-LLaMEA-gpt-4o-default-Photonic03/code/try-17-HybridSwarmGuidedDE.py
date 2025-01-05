import numpy as np

class HybridSwarmGuidedDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.swarm = np.random.uniform(0, 1, (self.population_size, dim))
        self.velocities = np.random.uniform(0, 1, (self.population_size, dim)) * 0.1
        self.best_positions = self.swarm.copy()
        self.global_best = None
        self.best_fitness = np.full(self.population_size, np.inf)
        self.global_best_fitness = np.inf
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                fitness = func(self.swarm[i])
                self.evaluations += 1

                if fitness < self.best_fitness[i]:
                    self.best_fitness[i] = fitness
                    self.best_positions[i] = self.swarm[i].copy()

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.swarm[i].copy()

                inertia_weight = 0.9 - 0.7 * (self.evaluations / self.budget)
                cognitive_coeff = 2.0 * np.random.rand()
                social_coeff = 2.0 * np.random.rand()

                cognitive_velocity = cognitive_coeff * (self.best_positions[i] - self.swarm[i])
                social_velocity = social_coeff * (self.global_best - self.swarm[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
                self.swarm[i] += self.velocities[i]
                self.swarm[i] = np.clip(self.swarm[i], lower_bound, upper_bound)

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.swarm[indices[0]], self.swarm[indices[1]], self.swarm[indices[2]]

                F = 0.5 + 0.5 * np.random.rand()  # Adaptive mutation factor
                mutant = np.clip(a + F * (b - c), lower_bound, upper_bound)
                crossover_rate = 0.9 - 0.4 * (self.evaluations / self.budget)
                crossover_indices = np.random.rand(self.dim) < crossover_rate
                trial = np.where(crossover_indices, mutant, self.swarm[i])

                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < self.best_fitness[i]:
                    self.swarm[i] = trial.copy()
                    self.best_positions[i] = trial.copy()
                    self.best_fitness[i] = trial_fitness
                    if trial_fitness < self.global_best_fitness:
                        self.global_best_fitness = trial_fitness
                        self.global_best = trial.copy()

        return self.global_best