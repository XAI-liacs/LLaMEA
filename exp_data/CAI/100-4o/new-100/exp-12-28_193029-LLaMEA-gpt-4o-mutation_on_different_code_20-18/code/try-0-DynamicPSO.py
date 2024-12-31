import numpy as np

class DynamicPSO:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia_max = 0.9
        self.inertia_min = 0.4
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component
        self.velocity_clamp = 0.2
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim)) * (ub - lb)
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.num_particles, np.Inf)

        global_best_position = None
        global_best_score = np.Inf

        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.num_particles):
                current_fitness = func(particles[i])
                evaluations += 1

                if current_fitness < personal_best_scores[i]:
                    personal_best_scores[i] = current_fitness
                    personal_best_positions[i] = particles[i]

                if current_fitness < global_best_score:
                    global_best_score = current_fitness
                    global_best_position = particles[i]

            if evaluations >= self.budget:
                break

            inertia = self.inertia_max - (self.inertia_max - self.inertia_min) * (evaluations / self.budget)
            for i in range(self.num_particles):
                cognitive_component = self.c1 * np.random.rand(self.dim) * (personal_best_positions[i] - particles[i])
                social_component = self.c2 * np.random.rand(self.dim) * (global_best_position - particles[i])
                velocities[i] = inertia * velocities[i] + cognitive_component + social_component
                velocities[i] = np.clip(velocities[i], -self.velocity_clamp, self.velocity_clamp)

                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

        self.f_opt = global_best_score
        self.x_opt = global_best_position

        return self.f_opt, self.x_opt