import numpy as np

class PSO_SA:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.num_particles = min(50, dim * 10)
        self.w = 0.9  # Increased initial inertia weight
        self.w_min = 0.4  # Added minimum inertia weight
        self.c1 = 1.5  # Cognitive component
        self.c2 = 2.0  # Increased social component
        self.temp = 1.0  # Initial temperature for SA

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in particles])

        global_best_score = np.min(personal_best_scores)
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]

        eval_count = self.num_particles

        while eval_count < self.budget:
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * r2 * (global_best_position - particles[i]))

                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

                f = func(particles[i])
                eval_count += 1

                if f < personal_best_scores[i] or np.exp((personal_best_scores[i] - f) / self.temp) > np.random.rand():
                    personal_best_positions[i] = np.copy(particles[i])
                    personal_best_scores[i] = f

                if f < global_best_score:
                    global_best_score = f
                    global_best_position = np.copy(particles[i])

                if eval_count >= self.budget:
                    break

            self.temp *= 0.95  # Faster cooling schedule
            self.w = max(self.w_min, self.w * 0.99)  # Decaying inertia weight

        self.f_opt, self.x_opt = global_best_score, global_best_position
        return self.f_opt, self.x_opt