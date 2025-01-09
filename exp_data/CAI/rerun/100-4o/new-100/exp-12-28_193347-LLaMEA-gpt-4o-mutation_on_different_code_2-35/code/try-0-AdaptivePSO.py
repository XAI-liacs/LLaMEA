import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=50):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.swarm_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        c1, c2 = 2.05, 2.05  # cognitive and social components
        w_max, w_min = 0.9, 0.4  # inertia weights
        iteration = 0

        while iteration < self.budget:
            for i in range(self.swarm_size):
                if iteration >= self.budget:
                    break
                f_val = func(particles[i])
                iteration += 1

                if f_val < personal_best_scores[i]:
                    personal_best_scores[i] = f_val
                    personal_best_positions[i] = particles[i]

                if f_val < global_best_score:
                    global_best_score = f_val
                    global_best_position = particles[i]

            if iteration >= self.budget:
                break

            w = w_max - ((w_max - w_min) * (iteration / self.budget))
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = c1 * r1 * (personal_best_positions[i] - particles[i])
                social_component = c2 * r2 * (global_best_position - particles[i])
                velocities[i] = w * velocities[i] + cognitive_component + social_component

                velocities[i] = np.clip(velocities[i], -1, 1)  # velocity clamping
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

        self.f_opt = global_best_score
        self.x_opt = global_best_position

        return self.f_opt, self.x_opt