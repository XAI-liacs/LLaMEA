import numpy as np

class AdaptiveParticleSwarm:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive (particle) weight
        self.c2 = 1.5  # social (swarm) weight
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, np.Inf)

        for _ in range(self.budget // self.num_particles):
            scores = np.array([func(pos) for pos in positions])
            for i, score in enumerate(scores):
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]
                if score < self.f_opt:
                    self.f_opt = score
                    self.x_opt = positions[i]

            global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
            
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (personal_best_positions[i] - positions[i])
                    + self.c2 * r2 * (global_best_position - positions[i])
                )
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)
            
            if np.random.rand() < 0.1:  # Random restart with 10% probability
                restart_indices = np.random.choice(self.num_particles, self.num_particles // 10, replace=False)
                positions[restart_indices] = np.random.uniform(lb, ub, (len(restart_indices), self.dim))

        return self.f_opt, self.x_opt