import numpy as np

class AdaptiveOBPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 5 + int(3 * np.log(dim))  # Adjusting population size based on dimensionality
        self.w = 0.9  # Inertia weight
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter

    def opposition_based_learning(self, position):
        return -position

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        lower_bound = -5.0
        upper_bound = 5.0

        # Initialize the swarm
        swarm = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.array([func(ind) for ind in personal_best_positions])

        # Determine the global best
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            # Calculate new velocities and update positions
            self.w = 0.4 + 0.5 * (self.budget - evaluations) / self.budget  # Dynamic inertia weight
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - swarm[i]) +
                                 self.c2 * r2 * (global_best_position - swarm[i]))
                
                swarm[i] = swarm[i] + velocities[i]

                # Clip the positions to the search space
                swarm[i] = np.clip(swarm[i], lower_bound, upper_bound)

                # Evaluate the new positions
                current_score = func(swarm[i])
                evaluations += 1
                if evaluations >= self.budget:
                    break

                # Opposition-based learning
                opposition_position = self.opposition_based_learning(swarm[i])
                opposition_position = np.clip(opposition_position, lower_bound, upper_bound)
                opposition_score = func(opposition_position)
                evaluations += 1
                if evaluations >= self.budget:
                    break

                # Update personal bests
                if opposition_score < current_score:
                    current_score = opposition_score
                    swarm[i] = opposition_position

                if current_score < personal_best_scores[i]:
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = swarm[i]

                # Update the global best
                if current_score < global_best_score:
                    global_best_score = current_score
                    global_best_position = swarm[i]

        return global_best_position, global_best_score