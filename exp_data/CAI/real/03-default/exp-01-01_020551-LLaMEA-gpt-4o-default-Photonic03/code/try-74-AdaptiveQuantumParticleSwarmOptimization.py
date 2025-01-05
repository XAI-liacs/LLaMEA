import numpy as np

class AdaptiveQuantumParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = max(10, 2 * dim)
        self.inertia_weight = 0.9
        self.cognitive_acceleration = 2.0
        self.social_acceleration = 2.0
        self.quantum_probability = 0.1
        self.inertia_damping = 0.99

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim)) * (ub - lb)
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([func(p) for p in positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index].copy()
        global_best_score = personal_best_scores[global_best_index]
        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_acceleration * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social_acceleration * r2 * (global_best_position - positions[i]))
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

                # Quantum-inspired position update
                if np.random.rand() < self.quantum_probability:
                    quantum_step = np.random.normal(0, 0.5, self.dim)
                    positions[i] = global_best_position + quantum_step * (ub - lb)
                    positions[i] = np.clip(positions[i], lb, ub)

                score = func(positions[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i].copy()

            # Update inertia weight
            self.inertia_weight *= self.inertia_damping

        return global_best_position, global_best_score