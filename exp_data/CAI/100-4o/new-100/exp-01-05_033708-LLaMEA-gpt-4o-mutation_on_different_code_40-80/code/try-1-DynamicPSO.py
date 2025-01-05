import numpy as np

class DynamicPSO:
    def __init__(self, budget=10000, dim=10, num_particles=50):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia_weight = 0.9
        self.c1 = 2.5  # Slightly increased cognitive component
        self.c2 = 1.5  # Slightly decreased social component
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-0.1, 0.1, (self.num_particles, self.dim))  # Random initial velocities
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, np.Inf)
        global_best_position = None

        for evals in range(self.budget):
            for i in range(self.num_particles):
                current_score = func(positions[i])
                if current_score < personal_best_scores[i]:
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = positions[i].copy()
                if current_score < self.f_opt:
                    self.f_opt = current_score
                    global_best_position = positions[i].copy()  # Update global best position

            self.inertia_weight = 0.9 - 0.4 * (evals / self.budget)  # Adjusted dynamic inertia

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                social_component = self.c2 * r2 * (global_best_position - positions[i])  # Use global best
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 cognitive_component + social_component)
                velocities[i] = np.clip(velocities[i], -0.2, 0.2)  # Velocity clamping
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)
        
        return self.f_opt, global_best_position