import numpy as np

class AdaptiveQuantumParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = max(10, dim * 2)
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.w_max = 0.9  # initial inertia weight
        self.w_min = 0.4  # final inertia weight
        self.beta = 0.3  # quantum update probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim)) * (ub - lb) / 10
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([func(p) for p in positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index].copy()
        global_best_score = personal_best_scores[global_best_index]
        evaluations = self.num_particles

        while evaluations < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (w * velocities[i] + 
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) + 
                                 self.c2 * r2 * (global_best_position - positions[i]))

                if np.random.rand() < self.beta:
                    q = np.random.normal(0, 0.1, self.dim)
                    positions[i] = global_best_position + q * (ub - lb)
                else:
                    positions[i] += velocities[i]
                    positions[i] = np.clip(positions[i], lb, ub)

                score = func(positions[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i].copy()

                if evaluations >= self.budget:
                    break

        return global_best_position, global_best_score