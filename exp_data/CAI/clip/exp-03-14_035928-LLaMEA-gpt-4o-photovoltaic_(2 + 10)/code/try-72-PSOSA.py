import numpy as np

class PSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.particles = np.random.rand(self.population_size, dim)
        self.velocities = np.random.rand(self.population_size, dim) - 0.5
        self.personal_best = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best = None
        self.global_best_score = np.inf
        self.iteration = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        scale = ub - lb
        inertia_weight = 0.9
        cognitive_constant = 1.7
        social_constant = 1.7
        temperature = 1.0
        temperature_decay = 0.99

        while self.iteration < self.budget:
            scores = np.array([func(p) for p in self.particles])
            for i in range(self.population_size):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best[i] = self.particles[i]

            best_particle_idx = np.argmin(scores)
            if scores[best_particle_idx] < self.global_best_score:
                self.global_best_score = scores[best_particle_idx]
                self.global_best = self.particles[best_particle_idx]

            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                adaptive_velocity = 1 - self.iteration / self.budget
                adaptive_cognitive = cognitive_constant + 0.5 * np.sin(self.iteration / self.budget * np.pi)
                adaptive_social = social_constant + 0.3 * (self.iteration / self.budget)
                
                # Integration of Lévy flight for improved exploration
                levy_flight = 0.01 * (np.random.normal(size=self.dim) * scale)
                self.velocities[i] = (
                    inertia_weight * self.velocities[i] +
                    adaptive_cognitive * r1 * (self.personal_best[i] - self.particles[i]) +
                    adaptive_social * r2 * (self.global_best - self.particles[i])
                ) * adaptive_velocity + levy_flight  # Adding Lévy flight
                
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lb, ub)

                # Temperature-based particle repositioning
                if np.random.rand() < np.exp(-scores[i] / temperature):
                    self.particles[i] = lb + np.random.rand(self.dim) * scale
                    scores[i] = func(self.particles[i])

            inertia_weight = 0.5 + 0.4 * (1 - self.iteration / self.budget)
            temperature *= temperature_decay
            self.iteration += self.population_size

        return self.global_best, self.global_best_score