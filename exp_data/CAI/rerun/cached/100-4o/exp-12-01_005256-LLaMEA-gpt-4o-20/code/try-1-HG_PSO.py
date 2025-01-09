import numpy as np

class HG_PSO:
    def __init__(self, budget, dim, swarm_size=50, inertia=0.7, cognitive=1.5, social=1.5, mutation_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.mutation_prob = mutation_prob
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility

        # Initialize positions and velocities
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))

        # Initialize personal bests and global best
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(x) for x in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.swarm_size
        inertia_decay = 0.99  # New: Decay rate for inertia
        min_inertia = 0.4  # New: Minimum inertia value

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    self.inertia * velocities[i] +
                    self.cognitive * r1 * (personal_best_positions[i] - positions[i]) +
                    self.social * r2 * (global_best_position - positions[i])
                )

                # Update position
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

                # Evaluate fitness
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

                if evaluations >= self.budget:
                    break

            # Dynamic inertia adjustment
            self.inertia = max(min_inertia, self.inertia * inertia_decay)  # New: Adjust inertia

            # Adaptive mutation
            success_rate = np.mean(personal_best_scores < global_best_score)  # New: Calculate success rate
            adaptive_mutation_prob = self.mutation_prob * (1 - success_rate)  # New: Adjust mutation probability
            mutation_mask = np.random.rand(self.swarm_size, self.dim) < adaptive_mutation_prob
            mutation_values = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
            positions = np.where(mutation_mask, mutation_values, positions)

        return global_best_position, global_best_score