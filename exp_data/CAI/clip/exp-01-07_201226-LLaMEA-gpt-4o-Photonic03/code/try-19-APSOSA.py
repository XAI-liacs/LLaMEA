import numpy as np

class APSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.iterations = budget // dim
        self.population_size = 30
        self.w_max = 0.9  # max inertia weight
        self.w_min = 0.4  # min inertia weight
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
        
        for iter_count in range(self.iterations):
            # Adaptive inertia weight
            self.w = self.w_max - iter_count * ((self.w_max - self.w_min) / self.iterations)
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions) * np.exp(-0.001 * iter_count))
            
            # Differential Evolution-inspired crossover
            rand_indices = np.random.randint(0, self.dim, self.dim)
            crossover_mask = np.random.rand(self.population_size, self.dim) < 0.5
            mutant_positions = positions[rand_indices] + velocities
            trial_positions = np.where(crossover_mask, mutant_positions, positions)
            positions = np.clip(trial_positions, lb, ub)
            
            scores = np.array([func(x) for x in positions])
            evals += self.population_size

            # Simulated Annealing acceptance criterion
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