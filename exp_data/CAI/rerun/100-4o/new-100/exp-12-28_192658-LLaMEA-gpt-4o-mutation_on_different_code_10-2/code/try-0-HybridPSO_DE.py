import numpy as np

class HybridPSO_DE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 100
        self.inertia_weight = 0.5
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.mutation_factor = 0.5
        self.crossover_prob = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(x) for x in personal_best_positions])

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_weight * r1 * (personal_best_positions[i] - population[i]) +
                                 self.social_weight * r2 * (global_best_position - population[i]))
                velocities[i] = np.clip(velocities[i], lb - ub, ub - lb)
                population[i] = np.clip(population[i] + velocities[i], lb, ub)

                if eval_count < self.budget:
                    eval_count += 1
                    score = func(population[i])
                    if score < personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = population[i].copy()
                        if score < global_best_score:
                            global_best_score = score
                            global_best_position = population[i].copy()

            # Adaptive Differential Evolution step
            for i in range(self.population_size):
                if eval_count < self.budget:
                    indices = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = population[np.random.choice(indices, 3, replace=False)]
                    mutant_vector = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                    cross_points = np.random.rand(self.dim) < self.crossover_prob
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True

                    trial_vector = np.where(cross_points, mutant_vector, population[i])
                    eval_count += 1
                    score = func(trial_vector)
                    if score < personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = trial_vector.copy()
                        if score < global_best_score:
                            global_best_score = score
                            global_best_position = trial_vector.copy()

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt