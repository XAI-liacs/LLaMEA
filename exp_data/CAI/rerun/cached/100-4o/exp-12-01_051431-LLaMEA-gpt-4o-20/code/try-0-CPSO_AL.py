import numpy as np

class CPSO_AL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = min(40, budget // dim)
        self.low_bound = -5.0
        self.up_bound = 5.0
        self.w = 0.7  # inertia weight
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.adaptive_lr_min = 0.4
        self.adaptive_lr_max = 0.9

    def _initialize_swarm(self):
        positions = np.random.uniform(self.low_bound, self.up_bound, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        return positions, velocities, personal_best_positions, personal_best_scores

    def _update_velocity(self, velocities, positions, personal_best_positions, global_best_position):
        r1, r2 = np.random.uniform(0, 1, (2, self.swarm_size, self.dim))
        cognitive_component = self.c1 * r1 * (personal_best_positions - positions)
        social_component = self.c2 * r2 * (global_best_position - positions)
        velocities = self.w * velocities + cognitive_component + social_component
        return velocities

    def __call__(self, func):
        positions, velocities, personal_best_positions, personal_best_scores = self._initialize_swarm()
        global_best_position = np.zeros(self.dim)
        global_best_score = np.inf
        eval_count = 0

        while eval_count < self.budget:
            # Evaluate current positions
            for i in range(self.swarm_size):
                if eval_count >= self.budget:
                    break
                score = func(positions[i])
                eval_count += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

            # Update velocities and positions
            velocities = self._update_velocity(velocities, positions, personal_best_positions, global_best_position)
            adaptive_lr = np.linspace(self.adaptive_lr_max, self.adaptive_lr_min, self.budget)
            adaptive_w = adaptive_lr[min(eval_count, self.budget - 1)]
            positions = np.clip(positions + adaptive_w * velocities, self.low_bound, self.up_bound)

        return global_best_position