import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, swarm_size=50, c1=2.0, c2=2.0, w_max=0.9, w_min=0.4):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.w_max = w_max  # Maximum inertia weight
        self.w_min = w_min  # Minimum inertia weight
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize swarm
        positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([func(x) for x in positions])
        
        # Initialize global best
        g_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[g_best_index]
        global_best_score = personal_best_scores[g_best_index]

        eval_count = self.swarm_size  # Initial evaluations done

        while eval_count < self.budget:
            # Update inertia weight
            w = self.w_max - ((self.w_max - self.w_min) * (eval_count / self.budget))

            for i in range(self.swarm_size):
                # Update velocity
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))

                # Update position
                positions[i] += velocities[i]
                # Ensure positions are within bounds
                positions[i] = np.clip(positions[i], func.bounds.lb, func.bounds.ub)

                # Evaluate new position
                f = func(positions[i])
                eval_count += 1

                # Update personal best
                if f < personal_best_scores[i]:
                    personal_best_scores[i] = f
                    personal_best_positions[i] = positions[i]

                # Update global best
                if f < global_best_score:
                    global_best_score = f
                    global_best_position = positions[i]

            # Early stopping if budget is exhausted
            if eval_count >= self.budget:
                break

        self.f_opt = global_best_score
        self.x_opt = global_best_position

        return self.f_opt, self.x_opt