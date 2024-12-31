import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        self.velocity_clamp = 0.2 * (5.0 - (-5.0))  # 20% of the search space range

    def __call__(self, func):
        # Initialize the swarm
        positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-self.velocity_clamp, self.velocity_clamp, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(x) for x in positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]

        evaluations = self.swarm_size

        while evaluations < self.budget:
            # Adaptive inertia weight
            w = self.w_max - ((self.w_max - self.w_min) * evaluations / self.budget)

            for i in range(self.swarm_size):
                # Update velocities
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    w * velocities[i] +
                    self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                    self.c2 * r2 * (global_best_position - positions[i])
                )
                velocities[i] = np.clip(velocities[i], -self.velocity_clamp, self.velocity_clamp)

                # Update positions
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], func.bounds.lb, func.bounds.ub)

                # Evaluate new position
                score = func(positions[i])
                evaluations += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

                # Stop if budget is reached
                if evaluations >= self.budget:
                    break

        return global_best_score, global_best_position