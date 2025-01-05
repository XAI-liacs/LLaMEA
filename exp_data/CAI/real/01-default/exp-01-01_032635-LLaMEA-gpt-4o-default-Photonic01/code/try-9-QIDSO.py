import numpy as np

class QIDSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = None
        self.velocities = None
        self.best_personal_positions = None
        self.best_personal_values = None
        self.global_best_position = None
        self.global_best_value = np.inf
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.5  # Social component
        self.inertia_weight = 0.7
        self.quantum_prob = 0.2
        self.swarm_size = min(40, budget)
    
    def initialize_particles(self, lb, ub):
        self.particles = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim)) * (ub - lb) * 0.1
        self.best_personal_positions = self.particles.copy()
        self.best_personal_values = np.full(self.swarm_size, np.inf)

    def quantum_position_update(self, position, best_position, lb, ub):
        direction = np.random.uniform(-1, 1, self.dim) * (best_position - position)
        new_position = position + direction * np.random.randn(self.dim)
        return np.clip(new_position, lb, ub)

    def dynamic_velocity_update(self, i):
        inertia = self.inertia_weight * self.velocities[i]
        cognitive = self.c1 * np.random.rand(self.dim) * (self.best_personal_positions[i] - self.particles[i])
        social = self.c2 * np.random.rand(self.dim) * (self.global_best_position - self.particles[i])
        return inertia + cognitive + social

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_particles(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                if np.random.rand() < self.quantum_prob:
                    new_position = self.quantum_position_update(self.particles[i], self.global_best_position, lb, ub)
                else:
                    self.velocities[i] = self.dynamic_velocity_update(i)
                    new_position = self.particles[i] + self.velocities[i]
                    new_position = np.clip(new_position, lb, ub)

                current_value = func(new_position)
                evaluations += 1

                if current_value < self.best_personal_values[i]:
                    self.best_personal_values[i] = current_value
                    self.best_personal_positions[i] = new_position.copy()

                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best_position = new_position.copy()

        return self.global_best_position, self.global_best_value