import numpy as np

class HybridPSO:
    def __init__(self, budget=10000, dim=10, swarm_size=50):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.c1 = 2.05  # Cognitive component
        self.c2 = 2.05  # Social component
        self.w = 0.7298  # Inertia weight

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.full(self.swarm_size, np.inf)

        for i in range(self.budget):
            if i % (self.budget // 100) == 0:  # Adaptively perturb every 1% of evaluations
                diff_factor = 0.8 + 0.2 * np.cos(np.pi * i / self.budget)
                for j in range(self.swarm_size):
                    if np.random.rand() < 0.3:  # 30% chance to apply perturbation
                        donor_vector = positions[j] + diff_factor * (personal_best_positions[j] - positions[j])
                        donor_vector = np.clip(donor_vector, lb, ub)
                        f_donor = func(donor_vector)
                        if f_donor < personal_best_values[j]:
                            personal_best_positions[j] = donor_vector
                            personal_best_values[j] = f_donor
                            if f_donor < self.f_opt:
                                self.f_opt = f_donor
                                self.x_opt = donor_vector

            # Evaluate fitness
            for j in range(self.swarm_size):
                current_value = func(positions[j])
                if current_value < personal_best_values[j]:
                    personal_best_positions[j] = positions[j]
                    personal_best_values[j] = current_value
                    if current_value < self.f_opt:
                        self.f_opt = current_value
                        self.x_opt = positions[j]

            # Update velocities and positions
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (self.x_opt - positions))
            positions += velocities
            positions = np.clip(positions, lb, ub)

        return self.f_opt, self.x_opt