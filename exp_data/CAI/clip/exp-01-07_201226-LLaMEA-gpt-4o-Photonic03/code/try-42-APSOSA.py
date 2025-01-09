import numpy as np

class APSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.iterations = budget // dim
        self.initial_population_size = 30
        self.final_population_size = 10
        self.w = 0.5  # initial inertia weight
        self.c1 = 2.05  # cognitive coefficient
        self.c2 = 2.05  # social coefficient
        self.initial_temp = 100  # initial temperature for simulated annealing
        self.final_temp = 1  # final temperature
        self.elite_percentage = 0.1  # elite percentage for elite-based criterion

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        temp = self.initial_temp
        positions = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(x) for x in personal_best_positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]
        evals = population_size

        for iter_num in range(self.iterations):
            # Dynamic population size
            population_size = int(self.initial_population_size - 
                                  (self.initial_population_size - self.final_population_size) * 
                                  (iter_num / self.iterations))
            r1, r2 = np.random.rand(2, population_size, self.dim)
            adaptive_w = self.w * (1 - iter_num / self.iterations)  # adaptive inertia weight
            velocities = (adaptive_w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            positions = np.clip(positions + velocities, lb, ub)
            
            scores = np.array([func(x) for x in positions])
            evals += population_size

            # Simulated Annealing acceptance criterion
            sorted_indices = np.argsort(scores)
            elite_count = max(1, int(self.elite_percentage * population_size))
            for i in sorted_indices[:elite_count]:  # apply acceptance only to elite
                if scores[i] < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_scores[i] = scores[i]
                    if scores[i] < global_best_score:
                        delta = scores[i] - global_best_score
                        probability = np.exp(-delta / temp)
                        if np.random.rand() < probability:
                            global_best_position = positions[i]
                            global_best_score = scores[i]

            # Adaptive cooling
            temp = self.initial_temp * (self.final_temp / self.initial_temp) ** (iter_num / self.iterations)

            if evals >= self.budget:
                break

        return global_best_position, global_best_score