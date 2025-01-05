import numpy as np

class QuantumInspiredDynamicAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1, self.c2 = 1.5, 1.5
        self.w_min, self.w_max = 0.4, 0.9
        self.success_rate_threshold = 0.2
        self.particles = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_fitness = np.inf
        self.success_rates = [0.5, 0.5]

    def levy_flight(self, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v) ** (1 / beta)
        return 0.01 * step

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        evaluations = 0
        inertia_weight = self.w_max

        while evaluations < self.budget:
            successes = [0, 0]
            for i in range(self.population_size):
                # Evaluate current fitness
                fitness = func(self.particles[i])
                evaluations += 1

                # Update personal best
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = self.particles[i]

                # Update global best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = self.particles[i]

            # Update inertia weight
            inertia_weight = self.w_max - ((self.w_max - self.w_min) * (evaluations / self.budget))

            for i in range(self.population_size):
                strategy = np.random.choice([0, 1], p=self.success_rates)
                r1, r2 = np.random.uniform(0, 1, 2)
                levy_step = self.levy_flight()

                if strategy == 0:
                    velocity = (inertia_weight * self.velocities[i] +
                                self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i]) +
                                self.c2 * r2 * (self.global_best_position - self.particles[i]) + levy_step)
                else:
                    velocity = (inertia_weight * self.velocities[i] +
                                self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i]) +
                                levy_step)

                self.velocities[i] = np.clip(velocity, lb - self.particles[i], ub - self.particles[i])
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lb, ub)

                # Evaluate success rate
                new_fitness = func(self.particles[i])
                evaluations += 1
                if new_fitness < fitness:
                    successes[strategy] += 1

            total_successes = sum(successes)
            if total_successes > 0:
                self.success_rates = [s / total_successes for s in successes]
                self.success_rates = [max(rate, self.success_rate_threshold) for rate in self.success_rates]

        return self.global_best_position