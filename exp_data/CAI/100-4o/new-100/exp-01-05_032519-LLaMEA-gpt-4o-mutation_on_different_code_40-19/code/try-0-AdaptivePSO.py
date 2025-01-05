import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub

        # Initialize particle positions and velocities
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.Inf)

        # Evaluate initial positions
        for i in range(self.swarm_size):
            score = func(positions[i])
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = positions[i].copy()
            if score < self.f_opt:
                self.f_opt = score
                self.x_opt = positions[i].copy()

        # Main PSO loop
        evals = self.swarm_size
        while evals < self.budget:
            w = 0.5 + np.random.rand() / 2  # Adaptive inertia weight
            c1, c2 = 1.5, 1.5  # Cognitive and Social coefficients

            for i in range(self.swarm_size):
                # Update velocity and position
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = c1 * r1 * (personal_best_positions[i] - positions[i])
                social = c2 * r2 * (self.x_opt - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social
                positions[i] += velocities[i]

                # Clip positions within the search space
                positions[i] = np.clip(positions[i], lb, ub)

                # Evaluate new position
                score = func(positions[i])
                evals += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()

                # Update global best
                if score < self.f_opt:
                    self.f_opt = score
                    self.x_opt = positions[i].copy()

                # Break if budget is exhausted
                if evals >= self.budget:
                    break

        return self.f_opt, self.x_opt