import numpy as np

class AdaptiveQuantumParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 10 + int(2 * np.sqrt(dim))
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.q_prob = 0.1  # Probability to use quantum-inspired velocity update

    def quantum_velocity_update(self, velocity, best_position, current_position, lb, ub):
        factor = np.random.uniform(0.5, 1.5, self.dim)
        new_velocity = factor * (best_position - current_position)
        return np.clip(new_velocity, lb - current_position, ub - current_position)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.swarm_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                # Evaluate current position
                score = func(positions[i])
                evaluations += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i].copy()

                # Update velocity
                inertia = self.w * velocities[i]
                cognitive = self.c1 * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i])
                social = self.c2 * np.random.rand(self.dim) * (global_best_position - positions[i])
                new_velocity = inertia + cognitive + social

                # Quantum-inspired velocity update
                if np.random.rand() < self.q_prob:
                    new_velocity = self.quantum_velocity_update(new_velocity, global_best_position, positions[i], lb, ub)

                velocities[i] = new_velocity

                # Update position
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

        return global_best_position, global_best_score