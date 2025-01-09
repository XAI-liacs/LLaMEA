import numpy as np

class PSO_ADM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30  # Population size
        self.w = 0.9  # Initial inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.c3 = 0.5  # Personal-best velocity component
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.velocity_limit = (self.upper_bound - self.lower_bound) * 0.1
        self.f = 0.5  # Differential mutation factor

    def __call__(self, func):
        # Initialize positions and velocities
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-self.velocity_limit, self.velocity_limit, (self.pop_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.pop_size, np.inf)

        # Evaluate initial population
        scores = np.apply_along_axis(func, 1, positions)
        evaluations = self.pop_size

        # Initialize global best
        global_best_index = np.argmin(scores)
        global_best_position = positions[global_best_index]
        global_best_score = scores[global_best_index]

        # Main optimization loop
        while evaluations < self.budget:
            # Dynamic inertia weight
            self.w = 0.9 - (0.5 * (evaluations / self.budget))
            for i in range(self.pop_size):
                # Update velocities
                r1, r2, r3 = np.random.rand(self.dim), np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = self.c2 * r2 * (global_best_position - positions[i])
                personal_velocity = self.c3 * r3 * (personal_best_positions[i] - global_best_position)
                velocities[i] = self.w * velocities[i] + cognitive_velocity + social_velocity + personal_velocity
                velocities[i] = np.clip(velocities[i], -self.velocity_limit, self.velocity_limit)

                # Update positions
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)
                
                # Apply adaptive differential mutation
                if np.random.rand() < 0.1:  # 10% chance to apply mutation
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    mutant = positions[indices[0]] + self.f * (positions[indices[1]] - positions[indices[2]])
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    if func(mutant) < scores[i]:
                        positions[i] = mutant

                # Evaluate new positions
                new_score = func(positions[i])
                evaluations += 1

                # Update personal best
                if new_score < personal_best_scores[i]:
                    personal_best_scores[i] = new_score
                    personal_best_positions[i] = positions[i]

                # Update global best
                if new_score < global_best_score:
                    global_best_score = new_score
                    global_best_position = positions[i]

                # Check if budget is exceeded
                if evaluations >= self.budget:
                    break

        return global_best_position, global_best_score