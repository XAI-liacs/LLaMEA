import numpy as np

class CollaborativeSwarmSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.c1 = 2.0
        self.c2 = 2.0
        self.inertia_weight = 0.5

    def __call__(self, func):
        positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(pos) for pos in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        eval_count = self.population_size
        neighborhood_size = 3  # Start with a small neighborhood

        while eval_count < self.budget:
            for i in range(self.population_size):
                self.inertia_weight = 0.4 + (0.5 * (1 - eval_count / self.budget))
                
                neighbors = np.random.choice(self.population_size, neighborhood_size, replace=False)
                local_best_position = min(neighbors, key=lambda idx: personal_best_scores[idx])
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.c1 * np.random.rand() * (personal_best_positions[local_best_position] - positions[i])
                    + self.c2 * np.random.rand() * (global_best_position - positions[i])
                )

                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], func.bounds.lb, func.bounds.ub)

                score = func(positions[i])
                eval_count += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

                if eval_count >= self.budget:
                    break

            neighborhood_size = min(neighborhood_size + 1, self.population_size)

        return global_best_position, global_best_score