import numpy as np

class QuantumLeapParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = max(10, dim * 2)
        self.c1 = 1.49618  # Cognitive coefficient
        self.c2 = 1.49618  # Social coefficient
        self.w = 0.7298    # Inertia weight
        self.quantum_rate = 0.1
        self.mutation_strength = 0.05

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim)) * (ub - lb)
        scores = np.array([func(pos) for pos in positions])
        personal_best_positions = positions.copy()
        personal_best_scores = scores.copy()
        global_best_index = np.argmin(scores)
        global_best_position = positions[global_best_index].copy()
        evaluations = self.swarm_size

        while evaluations < self.budget:
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            positions += velocities
            positions = np.clip(positions, lb, ub)

            if np.random.rand() < self.quantum_rate:
                quantum_jump = np.random.normal(0, self.mutation_strength, self.dim) * (ub - lb)
                positions += quantum_jump

            new_scores = np.array([func(pos) for pos in positions])
            evaluations += self.swarm_size

            for i in range(self.swarm_size):
                if new_scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = new_scores[i]
                    personal_best_positions[i] = positions[i].copy()
            
            new_global_best_index = np.argmin(new_scores)
            if new_scores[new_global_best_index] < scores[global_best_index]:
                global_best_index = new_global_best_index
                global_best_position = positions[global_best_index].copy()

        return global_best_position, personal_best_scores[global_best_index]