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

        while eval_count < self.budget:
            self.population_size = max(5, int(10 * (1 - eval_count / self.budget)))

            for i in range(self.population_size):
                self.inertia_weight = 0.4 + (0.5 * (1 - eval_count / self.budget))
                
                scaling_factor = np.random.rand()  
                velocities[i] = (
                    scaling_factor * self.inertia_weight * velocities[i] * 0.95  
                    + self.c1 * np.random.rand() * (personal_best_positions[i] - positions[i])
                    + self.c2 * np.random.rand() * (global_best_position - positions[i])
                )
                local_search_step = np.random.uniform(-0.2, 0.2, self.dim)  
                positions[i] = positions[i] + velocities[i] + local_search_step
                positions[i] = np.clip(positions[i], func.bounds.lb, func.bounds.ub)

                score = func(positions[i])
                eval_count += 1

                # Adaptive adjustment of cognitive and social components
                self.c1 = 1.5 + (1.5 * eval_count / self.budget)  # Increase cognitive component
                self.c2 = 2.5 - (1.5 * eval_count / self.budget)  # Decrease social component

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

                if eval_count >= self.budget:
                    break

        return global_best_position, global_best_score