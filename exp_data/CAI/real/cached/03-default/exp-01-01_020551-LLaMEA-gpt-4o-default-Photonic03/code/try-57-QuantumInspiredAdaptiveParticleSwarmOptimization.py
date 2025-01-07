import numpy as np

class QuantumInspiredAdaptiveParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = max(10, dim * 2)
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.quantum_swarm_size = self.swarm_size // 2
        self.inertia_weight = 0.9
        self.inertia_damping = 0.99
        self.quantum_factor = 0.05

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        position = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.swarm_size, self.dim)) * (ub - lb)
        personal_best_position = position.copy()
        scores = np.array([func(position[i]) for i in range(self.swarm_size)])
        personal_best_scores = scores.copy()
        global_best_index = np.argmin(scores)
        global_best_position = position[global_best_index].copy()
        evaluations = self.swarm_size

        while evaluations < self.budget:
            # Update velocities and positions
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            cognitive_velocity = self.c1 * r1 * (personal_best_position - position)
            social_velocity = self.c2 * r2 * (global_best_position - position)
            velocity = self.inertia_weight * velocity + cognitive_velocity + social_velocity
            position += velocity
            position = np.clip(position, lb, ub)

            # Evaluate new solutions
            scores = np.array([func(position[i]) for i in range(self.swarm_size)])
            evaluations += self.swarm_size

            # Update personal and global bests
            better_mask = scores < personal_best_scores
            personal_best_position[better_mask] = position[better_mask]
            personal_best_scores[better_mask] = scores[better_mask]
            if scores.min() < personal_best_scores[global_best_index]:
                global_best_index = scores.argmin()
                global_best_position = position[global_best_index]

            # Quantum-inspired swarm exploration
            if evaluations < self.budget:
                quantum_positions = global_best_position + np.random.normal(0, self.quantum_factor, (self.quantum_swarm_size, self.dim)) * (ub - lb)
                quantum_positions = np.clip(quantum_positions, lb, ub)
                quantum_scores = np.array([func(quantum_positions[i]) for i in range(self.quantum_swarm_size)])
                evaluations += self.quantum_swarm_size
                if quantum_scores.min() < personal_best_scores[global_best_index]:
                    global_best_index = quantum_scores.argmin()
                    global_best_position = quantum_positions[global_best_index]

            # Adaptive inertia update
            self.inertia_weight *= self.inertia_damping

        return global_best_position, personal_best_scores[global_best_index]