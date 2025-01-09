import numpy as np

class APSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.iterations = budget // dim
        self.population_size = 30
        self.w = 0.9  # Changed: adaptive inertia weight start
        self.c1 = 2.05  # cognitive coefficient
        self.c2 = 2.05  # social coefficient
        self.temp = 100  # initial temperature for simulated annealing
        self.cooling_rate = 0.99

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
        
        for i in range(self.iterations):
            self.w = 0.9 - (0.5 * i / self.iterations)  # Changed: adaptive inertia weight decrease
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions) * np.exp(-0.001 * i)) # Slight modification here
            positions = np.clip(positions + velocities, lb, ub)
            
            scores = np.array([func(x) for x in positions])
            evals += self.population_size

            # Simulated Annealing acceptance criterion
            for j in range(self.population_size):
                if scores[j] < personal_best_scores[j]:
                    personal_best_positions[j] = positions[j]
                    personal_best_scores[j] = scores[j]
                    if scores[j] < global_best_score:
                        delta = scores[j] - global_best_score
                        probability = np.exp(-delta / self.temp)
                        if np.random.rand() < probability:
                            global_best_position = positions[j]
                            global_best_score = scores[j]

            # Cool down the temperature
            self.temp *= self.cooling_rate

            if evals >= self.budget:
                break

        return global_best_position, global_best_score