import numpy as np

class SwarmLevyCooperativeOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.alpha = 0.1  # Influence of local best
        self.beta = 0.2   # Influence of global best
        self.levy_scale = 0.5
        self.history = []

    def levy_flight(self, step_size):
        u = np.random.normal(0, 1, self.dim)
        v = np.random.normal(0, 1, self.dim)
        return step_size * (u / np.abs(v) ** (1 / self.levy_scale))

    def __call__(self, func):
        self.bounds = func.bounds
        positions = np.random.uniform(self.bounds.lb, self.bounds.ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(ind) for ind in positions])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]

        self.history.extend(personal_best_scores)

        evaluations = self.swarm_size
        while evaluations < self.budget:
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (velocities[i] +
                                 self.alpha * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.beta * r2 * (global_best_position - positions[i]))
                
                positions[i] += velocities[i]

                # Apply Levy flight
                if np.random.rand() < 0.3:
                    positions[i] += self.levy_flight(0.1)

                positions[i] = np.clip(positions[i], self.bounds.lb, self.bounds.ub)
                score = func(positions[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                if score < personal_best_scores[global_best_idx]:
                    global_best_idx = i
                    global_best_position = personal_best_positions[i]

            self.history.extend(personal_best_scores)

        return global_best_position, personal_best_scores[global_best_idx], self.history