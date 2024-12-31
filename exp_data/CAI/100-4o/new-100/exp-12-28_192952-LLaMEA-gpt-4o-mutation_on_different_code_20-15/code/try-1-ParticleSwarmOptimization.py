import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30, inertia=0.9, cognitive=2.0, social=2.0):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, np.inf)
        global_best_position = None
        evaluations = 0
        inertia_weight_decay = (0.4 - 0.9) / self.budget

        while evaluations < self.budget:
            scores = np.array([func(pos) for pos in positions])
            evaluations += self.num_particles
            for i in range(self.num_particles):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = positions[i]

            best_particle_index = np.argmin(personal_best_scores)
            if personal_best_scores[best_particle_index] < self.f_opt:
                self.f_opt = personal_best_scores[best_particle_index]
                global_best_position = personal_best_positions[best_particle_index]
                self.x_opt = global_best_position

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia * velocities[i] +
                                 self.cognitive * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social * r2 * (global_best_position - positions[i]))
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)
                if np.any(positions[i] == lb) or np.any(positions[i] == ub):
                    velocities[i] *= -0.5  # Repulsion from boundaries

            self.inertia += inertia_weight_decay

        return self.f_opt, self.x_opt