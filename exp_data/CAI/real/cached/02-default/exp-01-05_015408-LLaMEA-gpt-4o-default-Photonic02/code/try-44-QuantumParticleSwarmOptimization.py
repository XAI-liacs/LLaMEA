import numpy as np

class QuantumParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 10 * dim
        self.alpha = 0.5
        self.beta = 0.5
        self.gamma = 1.5  # Attraction strength

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.rand(self.swarm_size, self.dim) * (ub - lb) + lb
        velocities = np.random.rand(self.swarm_size, self.dim) * 0.1 * (ub - lb)
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(ind) for ind in personal_best_positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Quantum potential-based velocity update
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.alpha * velocities[i]
                                 + self.beta * r1 * (personal_best_positions[i] - positions[i])
                                 + self.gamma * r2 * (global_best_position - positions[i]))

                # Update position
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)
                
                # Evaluate new position
                fitness = func(positions[i])
                evaluations += 1

                # Update personal best
                if fitness < personal_best_scores[i]:
                    personal_best_scores[i] = fitness
                    personal_best_positions[i] = positions[i]

                # Update global best
                if fitness < personal_best_scores[global_best_index]:
                    global_best_index = i
                    global_best_position = positions[i]

                if evaluations >= self.budget:
                    break

        return global_best_position, personal_best_scores[global_best_index]