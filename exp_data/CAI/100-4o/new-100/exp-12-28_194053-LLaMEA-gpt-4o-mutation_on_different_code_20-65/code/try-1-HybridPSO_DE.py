import numpy as np

class HybridPSO_DE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.9  # Updated initial inertia weight
        self.f = 0.5
        self.cr = 0.9
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(individual) for individual in population])
        global_best = np.min(personal_best_scores)
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - population[i]) +
                                 self.c2 * r2 * (global_best_position - population[i]))
                
                # Adaptive inertia weight
                self.w = 0.4 + 0.5 * (self.budget - evaluations) / self.budget
                
                candidate = population[i] + velocities[i]

                # Differential Evolution operation
                idxs = list(range(self.population_size))
                idxs.remove(i)
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = population[a] + self.f * (population[b] - population[c])
                cross_points = np.random.rand(self.dim) < self.cr
                candidate = np.where(cross_points, mutant, candidate)
                
                # Crowding distance for diversity
                candidate_score = func(candidate)
                if np.linalg.norm(candidate - population[i]) > np.linalg.norm(candidate - global_best_position):
                    candidate_score += 0.1  # Penalize less diverse solutions
                
                candidate = np.clip(candidate, lb, ub)
                candidate_score = func(candidate)
                evaluations += 1

                if candidate_score < personal_best_scores[i]:
                    personal_best_scores[i] = candidate_score
                    personal_best_positions[i] = candidate

                if candidate_score < global_best:
                    global_best = candidate_score
                    global_best_position = candidate

                if evaluations >= self.budget:
                    break

        self.f_opt = global_best
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt