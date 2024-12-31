import numpy as np

class CauchyParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.num_particles = int(np.sqrt(self.budget))  # A reasonable choice for the number of particles
        self.f_opt = np.inf
        self.x_opt = None
        self.w = 0.9  # Increased initial inertia weight for exploration
        self.w_min = 0.4  # Minimum inertia weight for exploitation
        self.c1 = 1.5  # Cognitive weight
        self.c2 = 1.5  # Social weight

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.num_particles, np.inf)

        global_best_score = np.inf
        global_best_position = None

        evals = 0
        while evals < self.budget:
            for i in range(self.num_particles):
                score = func(positions[i])
                evals += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

                if evals >= self.budget:
                    break

            # Update inertia weight based on progress
            self.w = self.w_min + (0.9 - self.w_min) * ((self.budget - evals) / self.budget)

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] + 
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))
                cauchy_step = np.random.standard_cauchy(self.dim) * 0.1
                positions[i] = np.clip(positions[i] + velocities[i] + cauchy_step, lb, ub)

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt