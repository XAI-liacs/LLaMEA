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
            # Adjust population size dynamically based on remaining budget
            self.population_size = max(5, int(10 * (1 - eval_count / self.budget)))

            for i in range(self.population_size):
                # Adjust inertia weight dynamically
                self.inertia_weight = 0.4 + (0.5 * (1 - eval_count / self.budget))
                
                # Update velocity with stochastic scaling
                scaling_factor = np.random.rand()  # Add stochastic scaling
                adaptive_lr = np.random.rand() * 0.1  # Adaptive learning rate
                velocities[i] = (
                    scaling_factor * self.inertia_weight * velocities[i]
                    + self.c1 * np.random.rand() * (personal_best_positions[i] - positions[i])
                    + self.c2 * adaptive_lr * np.random.rand() * (global_best_position - positions[i])
                )
                # Introduce adaptive local search with enhanced step size
                local_search_step = np.random.uniform(-0.2, 0.2, self.dim)
                positions[i] = positions[i] + velocities[i] + local_search_step
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