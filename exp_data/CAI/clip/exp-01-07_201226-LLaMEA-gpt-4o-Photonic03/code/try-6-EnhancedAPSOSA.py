import numpy as np

class EnhancedAPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.iterations = budget // dim
        self.initial_population_size = 30
        self.max_population_size = 50
        self.min_population_size = 20
        self.w_max = 0.9  # maximum inertia weight
        self.w_min = 0.4  # minimum inertia weight
        self.c1 = 2.05  # cognitive coefficient
        self.c2 = 2.05  # social coefficient
        self.temp = 100  # initial temperature for simulated annealing
        self.cooling_rate = 0.98  # slightly faster cooling

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        positions = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(x) for x in personal_best_positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]
        evals = population_size
        
        for iter in range(self.iterations):
            w = self.w_max - ((self.w_max - self.w_min) * (iter / self.iterations))
            r1, r2 = np.random.rand(2, population_size, self.dim)
            velocities = (w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            positions = np.clip(positions + velocities, lb, ub)
            
            scores = np.array([func(x) for x in positions])
            evals += population_size

            for i in range(population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_scores[i] = scores[i]
                    if scores[i] < global_best_score:
                        delta = scores[i] - global_best_score
                        probability = np.exp(-delta / self.temp)
                        if np.random.rand() < probability:
                            global_best_position = positions[i]
                            global_best_score = scores[i]

            self.temp *= self.cooling_rate
            
            population_size = int(self.min_population_size + (self.max_population_size - self.min_population_size) * (1 - iter / self.iterations))
            if evals >= self.budget:
                break

        return global_best_position, global_best_score