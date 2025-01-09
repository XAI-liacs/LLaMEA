import numpy as np

class EPSOSA_AL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.iterations = budget // dim
        self.population_size = 30
        self.temp = 100  # initial temperature for simulated annealing
        self.cooling_rate = 0.99
        self.w_min, self.w_max = 0.4, 0.9  # adaptive inertia weights
        self.c1_min, self.c1_max = 1.5, 2.5
        self.c2_min, self.c2_max = 1.5, 2.5

    def adaptive_learning_rate(self, iteration):
        fraction = iteration / self.iterations
        w = self.w_max - (self.w_max - self.w_min) * fraction
        c1 = self.c1_max - (self.c1_max - self.c1_min) * fraction
        c2 = self.c2_min + (self.c2_max - self.c2_min) * fraction
        return w, c1, c2

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
        
        for iter in range(self.iterations):
            w, c1, c2 = self.adaptive_learning_rate(iter)
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (w * velocities +
                          c1 * r1 * (personal_best_positions - positions) +
                          c2 * r2 * (global_best_position - positions))
            positions = np.clip(positions + velocities, lb, ub)
            
            scores = np.array([func(x) for x in positions])
            evals += self.population_size
            
            # Update personal and global bests
            for i in range(self.population_size):
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