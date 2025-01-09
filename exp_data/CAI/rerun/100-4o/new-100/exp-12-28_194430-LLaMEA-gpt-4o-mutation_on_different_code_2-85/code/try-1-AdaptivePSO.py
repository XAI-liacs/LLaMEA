import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        w = 0.9  # inertia weight
        c1 = 2.0  # cognitive coefficient
        c2 = 2.0  # social coefficient
        v_max = (self.bounds[1] - self.bounds[0]) * 0.1

        # Initialize particles
        positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.swarm_size, self.dim))
        velocities = np.random.uniform(-v_max, v_max, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.Inf)

        # Optimization loop
        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.swarm_size):
                f = func(positions[i])
                eval_count += 1
                if f < personal_best_scores[i]:
                    personal_best_scores[i] = f
                    personal_best_positions[i] = positions[i]
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = positions[i]

            # Update velocities and positions
            r1, r2 = np.random.rand(2, self.swarm_size, self.dim)
            velocities = (w * velocities +
                          c1 * r1 * (personal_best_positions - positions) +
                          c2 * r2 * (self.x_opt - positions))
            
            velocities = np.clip(velocities, -v_max, v_max)
            positions = np.clip(positions + velocities, self.bounds[0], self.bounds[1])

            # Adaptive inertia weight
            w = 0.5 + 0.4 * ((self.budget - eval_count) / self.budget)  # Adjusted formula

            # Random restart if stuck
            if eval_count % (self.budget // 5) == 0:
                stagnant = np.all(positions == personal_best_positions, axis=1)
                positions[stagnant] = np.random.uniform(self.bounds[0], self.bounds[1], (stagnant.sum(), self.dim))
                velocities[stagnant] = np.random.uniform(-v_max, v_max, (stagnant.sum(), self.dim))

        return self.f_opt, self.x_opt