import numpy as np

class PSO_SA_Optimizer:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.num_particles = 50
        self.f_opt = np.Inf
        self.x_opt = None
        self.w = 0.9  # Increased initial inertia weight
        self.c1 = 1.5
        self.c2 = 1.5
        self.temp = 1.0
        self.cooling_rate = 0.99
        self.success_rate = 0.1  # New attribute to track success rate

    def __call__(self, func):
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.num_particles, np.Inf)
        global_best_position = None
        global_best_score = np.Inf

        evaluations = 0

        while evaluations < self.budget:
            successful_updates = 0  # Counter for successful updates
            for i in range(self.num_particles):
                score = func(particles[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i].copy()
                    successful_updates += 1

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i].copy()

                if evaluations >= self.budget:
                    break

            self.success_rate = successful_updates / self.num_particles  # Update success rate
            self.w = 0.9 - 0.7 * self.success_rate  # Dynamic inertia weight adjustment

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] +
                                self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                self.c2 * r2 * (global_best_position - particles[i]))

                if np.random.rand() < np.exp(-score / self.temp):
                    velocities[i] *= 1.1
                else:
                    velocities[i] *= 0.9

                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], func.bounds.lb, func.bounds.ub)

            self.temp *= self.cooling_rate

        self.f_opt = global_best_score
        self.x_opt = global_best_position

        return self.f_opt, self.x_opt