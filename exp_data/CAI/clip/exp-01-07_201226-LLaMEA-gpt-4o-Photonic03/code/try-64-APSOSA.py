import numpy as np

class APSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.iterations = budget // dim
        self.initial_population_size = 30
        self.population_size = self.initial_population_size
        self.w = 0.5  # initial inertia weight
        self.c1 = 2.05  # cognitive coefficient
        self.c2 = 2.05  # social coefficient
        self.temp = 100  # initial temperature for simulated annealing
        self.cooling_rate = 0.99
        self.elite_percentage = 0.1  # elite percentage for elite-based criterion

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
            adaptive_w = self.w * (1 - iter_num / self.iterations)  # adaptive inertia weight
            velocities = (adaptive_w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            positions = np.clip(positions + velocities, lb, ub)

            scores = np.array([func(x) for x in positions])
            evals += self.population_size

            # Adaptive local search
            elite_indices = np.argsort(scores)[:int(self.elite_percentage * self.population_size)]
            for i in elite_indices:
                local_search_position = positions[i] + 0.1 * np.random.randn(self.dim)
                local_search_position = np.clip(local_search_position, lb, ub)
                local_search_score = func(local_search_position)
                evals += 1
                if local_search_score < scores[i]:
                    positions[i] = local_search_position
                    scores[i] = local_search_score

            # Simulated Annealing acceptance criterion
            for i in elite_indices:
                if scores[i] < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_scores[i] = scores[i]
                    if scores[i] < global_best_score:
                        delta = scores[i] - global_best_score
                        probability = np.exp(-delta / self.temp)
                        if np.random.rand() < probability:
                            global_best_position = positions[i]
                            global_best_score = scores[i]

            # Dyn. population resizing
            if iter_num % 10 == 0 and self.population_size > 5:
                self.population_size -= 1
                positions = positions[:self.population_size]
                velocities = velocities[:self.population_size]
                personal_best_positions = personal_best_positions[:self.population_size]
                personal_best_scores = personal_best_scores[:self.population_size]

            # Cool down the temperature
            self.temp *= self.cooling_rate

            if evals >= self.budget:
                break

        return global_best_position, global_best_score