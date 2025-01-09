import numpy as np

class APSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.iterations = budget // dim
        self.population_size = 30
        self.w = 0.5
        self.c1 = 2.05
        self.c2 = 2.05
        self.temp = 100
        self.cooling_rate = 0.95  # Adjusted cooling rate
        self.elite_percentage = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(x) for x in personal_best_positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]
        evals = self.population_size

        for iter_num in range(self.iterations):
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            adaptive_w = self.w * (1 - iter_num / self.iterations)
            velocities = (adaptive_w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            positions = np.clip(positions + velocities, lb, ub)
            
            scores = np.array([func(x) for x in positions])
            evals += self.population_size

            # Dynamic neighborhood search
            for i in range(self.population_size):
                neighbors = np.random.choice(range(self.population_size), size=5, replace=False)
                best_neighbor = min(neighbors, key=lambda x: personal_best_scores[x])
                if personal_best_scores[best_neighbor] < personal_best_scores[i]:
                    personal_best_positions[i] = personal_best_positions[best_neighbor]
                    personal_best_scores[i] = personal_best_scores[best_neighbor]

            # Simulated Annealing acceptance criterion
            sorted_indices = np.argsort(scores)
            elite_count = max(1, int(self.elite_percentage * self.population_size))
            for i in sorted_indices[:elite_count]:
                if scores[i] < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_scores[i] = scores[i]
                    if scores[i] < global_best_score:
                        delta = scores[i] - global_best_score
                        probability = np.exp(-delta / self.temp)
                        if np.random.rand() < probability:
                            global_best_position = positions[i]
                            global_best_score = scores[i]

            # Cool down the temperature
            self.temp *= self.cooling_rate

            if evals >= self.budget:
                break

        return global_best_position, global_best_score