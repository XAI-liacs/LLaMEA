import numpy as np

class QuantumInspiredDynamicSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 60
        self.q_factor = 0.8
        self.c1, self.c2 = 1.5, 1.5  # cognitive and social components
        self.inertia_weight = 0.7
        self.velocity_clamp = 0.1 * (dim ** 0.5)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        position_quantum = np.random.uniform(0, 1, (self.swarm_size, self.dim))
        positions = lb + (ub - lb) * np.cos(np.pi * position_quantum)
        velocities = np.random.uniform(-self.velocity_clamp, self.velocity_clamp, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.array([func(x) for x in positions])
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]

        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i]
                                 + self.c1 * r1 * (personal_best_positions[i] - positions[i])
                                 + self.c2 * r2 * (global_best_position - positions[i]))
                velocities[i] = np.clip(velocities[i], -self.velocity_clamp, self.velocity_clamp)

                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

                quantum_shift = self.q_factor * (np.random.rand(self.dim) - 0.5)
                positions[i] += quantum_shift
                positions[i] = np.clip(positions[i], lb, ub)

                fitness = func(positions[i])
                evaluations += 1

                if fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_fitness[i] = fitness
                    if fitness < personal_best_fitness[global_best_idx]:
                        global_best_idx = i
                        global_best_position = positions[i]

        return global_best_position