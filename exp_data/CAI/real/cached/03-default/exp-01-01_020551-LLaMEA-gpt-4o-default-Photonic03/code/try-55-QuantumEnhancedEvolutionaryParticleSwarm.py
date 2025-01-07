import numpy as np

class QuantumEnhancedEvolutionaryParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = max(20, 5 * dim)
        self.cognitive_coef = 2.0
        self.social_coef = 2.0
        self.quantum_coef = 0.2
        self.inertia_weight = 0.7
        self.inertia_damping = 0.99

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([func(pos) for pos in personal_best_positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index].copy()
        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_velocity = self.cognitive_coef * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = self.social_coef * r2 * (global_best_position - positions[i])
                velocities[i] = self.inertia_weight * velocities[i] + cognitive_velocity + social_velocity

                # Quantum-inspired update
                if np.random.rand() < self.quantum_coef:
                    q = np.random.normal(0, 0.5)
                    quantum_velocity = q * (ub - lb) * (global_best_position - positions[i])
                    velocities[i] += quantum_velocity

                # Update particle position
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

                score = func(positions[i])
                evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i].copy()
                    personal_best_scores[i] = score
                    if score < personal_best_scores[global_best_index]:
                        global_best_index = i
                        global_best_position = personal_best_positions[i].copy()

            self.inertia_weight *= self.inertia_damping

        return global_best_position, personal_best_scores[global_best_index]