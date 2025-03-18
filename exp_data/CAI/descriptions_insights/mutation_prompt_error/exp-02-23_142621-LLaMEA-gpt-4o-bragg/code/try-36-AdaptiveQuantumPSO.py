import numpy as np

class AdaptiveQuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.0
        self.c2 = 2.0
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
        chaotic_map = np.mod(np.arange(1, self.population_size + 1) * 0.7, 1)
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

                # Chaos-enhanced velocity update
                chaos_factor = 1 - chaotic_map[i]
                self.velocities[i] = (self.inertia * self.velocities[i] + 
                                      self.c1 * u * (pb - self.particles[i]) + 
                                      self.c2 * (1 - u) * (gb - self.particles[i])) * chaos_factor
                self.particles[i] += self.velocities[i]

                if np.random.rand() < self.mutation_rate:
                    diversity = np.std(self.particles, axis=0).mean()
                    levy_flight_step = np.random.standard_cauchy(size=self.dim) * (0.01 + diversity)
                    self.particles[i] += levy_flight_step * (budget_remaining / self.budget)

                if np.random.rand() < 0.1:
                    diversity = np.std(self.particles, axis=0).mean()
                    self.particles[i] += 0.01 * np.cos(np.linspace(0, np.pi, self.dim)) * diversity

            # Adaptive inertia update
            self.inertia = max(0.4, self.inertia * self.inertia_decay * (budget_remaining / self.budget))
            self.mutation_rate = self.min_mutation_rate + (
                (self.max_mutation_rate - self.min_mutation_rate) * (budget_remaining / self.budget))
            dynamic_population_size = max(10, dynamic_population_size - 1)

        return self.global_best_position, self.global_best_score