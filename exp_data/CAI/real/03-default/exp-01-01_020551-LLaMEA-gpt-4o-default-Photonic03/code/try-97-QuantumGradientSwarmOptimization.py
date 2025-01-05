import numpy as np

class QuantumGradientSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = max(10, dim * 2)
        self.alpha = 0.5  # Quantum influence factor
        self.beta = 0.1  # Gradient influence factor
        self.omega = 0.5  # Inertia weight
        self.phi_p = 0.5  # Personal attraction
        self.phi_g = 0.5  # Global attraction

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm_positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        swarm_velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = swarm_positions.copy()
        scores = np.array([func(swarm_positions[i]) for i in range(self.swarm_size)])
        personal_best_scores = scores.copy()
        global_best_index = np.argmin(scores)
        global_best_position = personal_best_positions[global_best_index].copy()
        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                gradient = np.zeros(self.dim)
                for j in range(self.dim):
                    delta = (ub[j] - lb[j]) * 1e-8
                    original_value = swarm_positions[i, j]
                    swarm_positions[i, j] = original_value + delta
                    score_up = func(swarm_positions[i])
                    swarm_positions[i, j] = original_value - delta
                    score_down = func(swarm_positions[i])
                    swarm_positions[i, j] = original_value
                    gradient[j] = (score_up - score_down) / (2 * delta)
                evaluations += 2 * self.dim

                swarm_velocities[i] = (
                    self.omega * swarm_velocities[i]
                    + self.phi_p * np.random.rand(self.dim) * (personal_best_positions[i] - swarm_positions[i])
                    + self.phi_g * np.random.rand(self.dim) * (global_best_position - swarm_positions[i])
                    - self.beta * gradient
                    + self.alpha * np.random.normal(0, 1, self.dim)
                )
                swarm_positions[i] = np.clip(swarm_positions[i] + swarm_velocities[i], lb, ub)
                new_score = func(swarm_positions[i])
                evaluations += 1

                if new_score < personal_best_scores[i]:
                    personal_best_positions[i] = swarm_positions[i]
                    personal_best_scores[i] = new_score
                    if new_score < personal_best_scores[global_best_index]:
                        global_best_position = swarm_positions[i]
                        global_best_index = i

        return global_best_position, personal_best_scores[global_best_index]