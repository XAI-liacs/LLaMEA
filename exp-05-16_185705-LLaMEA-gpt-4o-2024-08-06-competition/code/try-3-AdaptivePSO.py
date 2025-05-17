import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.w_min = 0.4  # Minimum inertia weight
        self.w_max = 0.9  # Maximum inertia weight
        self.c1 = 2.0     # Cognitive coefficient
        self.c2 = 2.0     # Social coefficient

    def __call__(self, func):
        # Initialize swarm
        positions = np.random.uniform(-100, 100, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(pos) for pos in positions])
        self.f_opt = np.min(personal_best_scores)
        self.x_opt = personal_best_positions[np.argmin(personal_best_scores)]
        
        global_best_position = self.x_opt
        global_best_score = self.f_opt

        eval_count = self.swarm_size
        
        while eval_count < self.budget:
            # Adaptive inertia weight
            w = self.w_max - (self.w_max - self.w_min) * (eval_count / self.budget)

            for i in range(self.swarm_size):
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))
                
                # Update position
                positions[i] += velocities[i]

                # Enforce search space boundaries
                positions[i] = np.clip(positions[i], -100, 100)

                # Evaluate new position
                score = func(positions[i])
                eval_count += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

                # Early stopping if budget reached
                if eval_count >= self.budget:
                    break
            
            self.f_opt = global_best_score
            self.x_opt = global_best_position

        return self.f_opt, self.x_opt