import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30, c1=2.0, c2=2.0, w=0.9, w_decay=0.99, w_min=0.4):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.w_decay = w_decay
        self.w_min = w_min
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub

        positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))

        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.num_particles, np.Inf)

        global_best_position = None
        global_best_score = np.Inf

        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.num_particles):
                if evaluations >= self.budget:
                    break
                
                current_position = positions[i]
                score = func(current_position)
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = current_position

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = current_position

            r1, r2 = np.random.rand(2)
            velocities = self.w * velocities + self.c1 * r1 * (personal_best_positions - positions) + self.c2 * r2 * (global_best_position - positions)
            positions = np.clip(positions + velocities, lb, ub)
            self.w = max(self.w_min, self.w * self.w_decay)  # Adjust inertia weight

        self.f_opt = global_best_score
        self.x_opt = global_best_position

        return self.f_opt, self.x_opt