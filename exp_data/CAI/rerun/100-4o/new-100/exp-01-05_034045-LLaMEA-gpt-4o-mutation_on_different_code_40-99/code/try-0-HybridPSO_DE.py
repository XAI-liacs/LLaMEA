import numpy as np

class HybridPSO_DE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 30
        self.w = 0.5  # Inertia weight for PSO
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.5  # Social component
        self.F = 0.8  # Differential evolution scaling factor
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        np.random.seed()
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(x) for x in population])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = np.copy(personal_best_positions[global_best_index])

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # PSO step
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - population[i]) +
                                 self.c2 * r2 * (global_best_position - population[i]))
                candidate_solution = population[i] + velocities[i]

                if evaluations < self.budget:
                    candidate_score = func(candidate_solution)
                    evaluations += 1
                else:
                    break

                if candidate_score < personal_best_scores[i]:
                    personal_best_scores[i] = candidate_score
                    personal_best_positions[i] = candidate_solution

                if candidate_score < self.f_opt:
                    self.f_opt = candidate_score
                    self.x_opt = candidate_solution

                # DE step
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = a + self.F * (b - c)
                trial_vector = np.where(np.random.rand(self.dim) < self.CR, mutant_vector, population[i])

                if evaluations < self.budget:
                    trial_score = func(trial_vector)
                    evaluations += 1
                else:
                    break

                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial_vector
                    population[i] = trial_vector

                if trial_score < self.f_opt:
                    self.f_opt = trial_score
                    self.x_opt = trial_vector

            global_best_index = np.argmin(personal_best_scores)
            global_best_position = np.copy(personal_best_positions[global_best_index])

        return self.f_opt, self.x_opt