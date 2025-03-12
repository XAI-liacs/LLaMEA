import numpy as np

class CollaborativeSwarmSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10  # Number of particles in the swarm
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component
        self.inertia_weight = 0.9  # Initial inertia weight

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
                # Update inertia weight with nonlinear decay
                self.inertia_weight = 0.9 - 0.4 * (eval_count / self.budget)**2  
                # Update velocity
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.c1 * np.random.rand() * (personal_best_positions[i] - positions[i])
                    + self.c2 * np.random.rand() * (global_best_position - positions[i])
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

                # Dynamically adjust population size
                if eval_count % (self.budget // 2) == 0:
                    if global_best_score > 0.1:
                        self.population_size = min(20, self.population_size + 1)

                # Check if budget is exhausted
                if eval_count >= self.budget:
                    break

        return global_best_position, global_best_score