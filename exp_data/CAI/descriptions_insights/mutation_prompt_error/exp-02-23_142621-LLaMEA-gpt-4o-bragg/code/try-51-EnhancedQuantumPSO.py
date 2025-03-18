import numpy as np

class EnhancedQuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5
        self.c2 = 1.5
        self.alpha = 0.75
        self.inertia = 0.9
        self.inertia_decay = 0.995
        self.mutation_rate = 0.15
        self.max_mutation_rate = 0.3
        self.min_mutation_rate = 0.05
        self.particles = np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.rand(self.population_size, self.dim) - 0.5
        self.best_particle_positions = np.copy(self.particles)
        self.best_particle_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def __call__(self, func):
        budget_remaining = self.budget
        dynamic_population_size = self.population_size

        # Chaotic map initialization
        chaotic_map = (np.sin(np.arange(1, self.population_size + 1) ** 2) * 0.5 + 0.5)
        self.particles = func.bounds.lb + chaotic_map[:, None] * (func.bounds.ub - func.bounds.lb)

        while budget_remaining > 0:
            for i in range(dynamic_population_size):
                self.particles[i] = np.clip(self.particles[i], func.bounds.lb, func.bounds.ub)
                
                score = func(self.particles[i])
                budget_remaining -= 1

                if score < self.best_particle_scores[i]:
                    self.best_particle_scores[i] = score
                    self.best_particle_positions[i] = self.particles[i].copy()

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i].copy()
                
                if budget_remaining <= 0:
                    break

            for i in range(dynamic_population_size):
                pb = self.best_particle_positions[i]
                gb = self.global_best_position
                u = np.random.rand(self.dim)

                # Chaos-enhanced velocity update with adaptive learning rates
                learning_rate = 1.0 / (1.0 + np.exp(-self.best_particle_scores[i]))
                self.velocities[i] = (self.inertia * self.velocities[i] +
                                      self.c1 * u * (pb - self.particles[i]) * learning_rate +
                                      self.c2 * (1 - u) * (gb - self.particles[i]) * learning_rate)
                self.particles[i] += self.velocities[i]

                # Adaptive differential evolution with Lévy flight
                if np.random.rand() < self.mutation_rate:
                    idxs = np.random.choice(dynamic_population_size, 3, replace=False)
                    a, b, c = self.particles[idxs]
                    step = np.random.normal(0, 1, size=self.dim)
                    levy_step = a + 0.8 * (b - c) + 0.01 * step
                    self.particles[i] = np.clip(levy_step, func.bounds.lb, func.bounds.ub)

                # Dynamic clustering adjustment
                if np.random.rand() < 0.1:
                    center = np.mean(self.particles, axis=0)
                    self.particles[i] += 0.02 * (center - self.particles[i])

            # Adaptive inertia update
            self.inertia = max(0.4, self.inertia * self.inertia_decay * (budget_remaining / self.budget))
            self.mutation_rate = self.min_mutation_rate + (
                (self.max_mutation_rate - self.min_mutation_rate) * (budget_remaining / self.budget))
            dynamic_population_size = max(10, dynamic_population_size - 1)

        return self.global_best_position, self.global_best_score