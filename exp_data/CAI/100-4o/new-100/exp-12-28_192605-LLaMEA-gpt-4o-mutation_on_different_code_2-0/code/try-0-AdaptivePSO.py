import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = swarm.copy()
        personal_best_scores = np.full(self.num_particles, np.inf)

        for i in range(self.num_particles):
            score = func(swarm[i])
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = swarm[i].copy()

            if score < self.f_opt:
                self.f_opt = score
                self.x_opt = swarm[i].copy()

        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]

        iteration = 0
        inertia_weight = 0.9
        while iteration < self.budget:
            inertia_weight = 0.9 - (0.5 * (iteration / self.budget))  # adaptive inertia weight
            for i in range(self.num_particles):
                velocities[i] = (inertia_weight * velocities[i] +
                                 np.random.uniform(0, 1) * (personal_best_positions[i] - swarm[i]) +
                                 np.random.uniform(0, 1) * (global_best_position - swarm[i]))
                swarm[i] = np.clip(swarm[i] + velocities[i], lb, ub)

                score = func(swarm[i])
                iteration += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = swarm[i].copy()

                if score < self.f_opt:
                    self.f_opt = score
                    self.x_opt = swarm[i].copy()
                
                if iteration >= self.budget:
                    break

            global_best_position = personal_best_positions[np.argmin(personal_best_scores)]

        return self.f_opt, self.x_opt