import numpy as np

class CollaborativeSwarmSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10  # Number of particles in the swarm
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component
        self.inertia_weight = 0.5  # Inertia weight for velocity update

    def __call__(self, func):
        # Initialize positions and velocities
        positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(pos) for pos in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                # Adjust inertia weight dynamically
                self.inertia_weight = 0.4 + (0.5 * (1 - eval_count / self.budget))
                
                # Update cognitive and social components dynamically
                adaptive_c1 = self.c1 - (1.5 * (eval_count / self.budget))
                adaptive_c2 = self.c2 + (1.5 * (eval_count / self.budget))
                
                # Update velocity with stochastic scaling
                scaling_factor = np.random.rand()  # Add stochastic scaling
                velocities[i] = (
                    scaling_factor * self.inertia_weight * velocities[i]  # Apply scaling factor
                    + adaptive_c1 * np.random.rand() * (personal_best_positions[i] - positions[i])
                    + adaptive_c2 * np.random.rand() * (global_best_position - positions[i])
                )
                # Update position
                positions[i] = positions[i] + velocities[i]
                # Clamp the positions within bounds
                positions[i] = np.clip(positions[i], func.bounds.lb, func.bounds.ub)

                # Evaluate the new position
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

                # Check if budget is exhausted
                if eval_count >= self.budget:
                    break

        return global_best_position, global_best_score