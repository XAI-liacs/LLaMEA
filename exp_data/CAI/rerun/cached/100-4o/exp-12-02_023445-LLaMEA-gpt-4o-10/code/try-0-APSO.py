import numpy as np

class APSO:
    def __init__(self, budget, dim, lower_bound=-5.0, upper_bound=5.0, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.swarm_size = swarm_size
        self.c1 = 2.0  # cognitive component
        self.c2 = 2.0  # social component
        self.w_max = 0.9
        self.w_min = 0.4
        self.v_max = (upper_bound - lower_bound) * 0.2
        self.iterations = budget // swarm_size

    def __call__(self, func):
        # Initialize swarm
        positions = self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.rand(self.swarm_size, self.dim)
        velocities = np.random.uniform(-self.v_max, self.v_max, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) for p in personal_best_positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]

        evals = self.swarm_size

        for i in range(self.iterations):
            w = self.w_max - (self.w_max - self.w_min) * (i / self.iterations)
            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)

            velocities = (w * velocities +
                self.c1 * r1 * (personal_best_positions - positions) +
                self.c2 * r2 * (global_best_position - positions))

            # Ensuring velocities are within limits
            np.clip(velocities, -self.v_max, self.v_max, out=velocities)

            # Update positions
            positions += velocities
            np.clip(positions, self.lower_bound, self.upper_bound, out=positions)

            # Evaluate new solutions
            scores = np.array([func(p) for p in positions])
            evals += self.swarm_size

            # Update personal bests
            better_mask = scores < personal_best_scores
            personal_best_scores = np.where(better_mask, scores, personal_best_scores)
            personal_best_positions = np.where(better_mask[:, np.newaxis], positions, personal_best_positions)

            # Update global best
            current_best_index = np.argmin(personal_best_scores)
            current_best_score = personal_best_scores[current_best_index]
            if current_best_score < global_best_score:
                global_best_score = current_best_score
                global_best_position = personal_best_positions[current_best_index]

            if evals >= self.budget:
                break

        return global_best_position, global_best_score