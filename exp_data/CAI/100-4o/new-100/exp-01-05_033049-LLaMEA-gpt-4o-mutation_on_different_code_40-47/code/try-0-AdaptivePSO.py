import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        particles = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.swarm_size, np.Inf)

        for _ in range(self.budget // self.swarm_size):
            for i, particle in enumerate(particles):
                score = func(particle)
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particle
                if score < self.f_opt:
                    self.f_opt = score
                    self.x_opt = particle

            global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
            inertia_weight = 0.5 + np.random.rand() / 2.0
            cognitive_coeff = 1.5 + np.random.rand()
            social_coeff = 1.5 + np.random.rand()

            velocities = inertia_weight * velocities + \
                         cognitive_coeff * np.random.rand(self.swarm_size, self.dim) * (personal_best_positions - particles) + \
                         social_coeff * np.random.rand(self.swarm_size, self.dim) * (global_best_position - particles)

            particles += velocities
            particles = np.clip(particles, lb, ub)

        return self.f_opt, self.x_opt