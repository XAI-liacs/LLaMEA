import numpy as np

class QuantumInspiredAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = max(10, 2 * dim)
        self.beta = 0.3  # Quantum exploration parameter
        self.c1, self.c2 = 2.0, 2.0  # Cognitive and social coefficients
        self.inertia_weight = 0.7
        self.inertia_decay = 0.99  # Decay for inertia weight
        self.velocity_clamp = 0.1  # Clamp for velocity

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub

        # Initialize swarm
        position = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocity = np.random.uniform(-self.velocity_clamp, self.velocity_clamp, (self.swarm_size, self.dim))
        personal_best_position = position.copy()
        personal_best_score = np.array([func(pos) for pos in position])
        global_best_index = np.argmin(personal_best_score)
        global_best_position = personal_best_position[global_best_index].copy()

        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Update velocities
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocity[i] = (self.inertia_weight * velocity[i] +
                               self.c1 * r1 * (personal_best_position[i] - position[i]) +
                               self.c2 * r2 * (global_best_position - position[i]))

                # Clamp velocity
                velocity[i] = np.clip(velocity[i], -self.velocity_clamp, self.velocity_clamp)

                # Update positions
                position[i] += velocity[i]

                # Quantum-inspired exploration
                if np.random.rand() < self.beta:
                    q = np.random.normal(0, 0.5, self.dim)  # Quantum step
                    position[i] = global_best_position + q * (ub - lb)

                # Boundary handling
                position[i] = np.clip(position[i], lb, ub)

                # Evaluate new positions
                score = func(position[i])
                evaluations += 1

                # Update personal bests
                if score < personal_best_score[i]:
                    personal_best_position[i] = position[i].copy()
                    personal_best_score[i] = score

            # Update global best
            current_best_index = np.argmin(personal_best_score)
            if personal_best_score[current_best_index] < personal_best_score[global_best_index]:
                global_best_index = current_best_index
                global_best_position = personal_best_position[global_best_index].copy()

            # Adapt inertia weight
            self.inertia_weight *= self.inertia_decay

        return global_best_position, personal_best_score[global_best_index]