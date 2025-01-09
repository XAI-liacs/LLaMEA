import numpy as np

class APSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.iterations = budget // dim
        self.population_size = 30
        self.w = 0.9  # initial inertia weight
        self.w_min = 0.4  # minimum inertia weight
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

        for iteration in range(self.iterations):
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            self.w = self.w_min + (0.9 - self.w_min) * (1 - iteration / self.iterations)  # linearly decreasing inertia weight
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))

            # Differential Evolution mutation
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                mutant = personal_best_positions[indices[0]] + 0.8 * (personal_best_positions[indices[1]] - personal_best_positions[indices[2]])
                mutant = np.clip(mutant, lb, ub)
                if func(mutant) < personal_best_scores[i]:
                    positions[i] = mutant

            positions = np.clip(positions + velocities, lb, ub)
            scores = np.array([func(x) for x in positions])
            evals += self.population_size

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

            self.temp *= self.cooling_rate

            if evals >= self.budget:
                break

        return global_best_position, global_best_score