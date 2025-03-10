import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.inertia_weight = 0.9  # Increased initial inertia for better exploration
        self.cognitive_coef = 1.5
        self.social_coef = 1.7
        self.temperature = 1000
        self.cooling_rate = 0.95  # Modified cooling rate for adaptive cooling
        self.min_inertia_weight = 0.3  # Adjusted minimum inertia weight
        self.adaptive_perturbation = 0.05  # New adaptive perturbation factor

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)
        evaluations = self.population_size

        while evaluations < self.budget:
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive_term = self.cognitive_coef * r1 * (personal_best_positions - population)
            social_term = self.social_coef * r2 * (global_best_position - population)
            velocities = self.inertia_weight * velocities + cognitive_term + social_term
            population += 0.6 * velocities  # Adjusted position update for improved balance
            population = np.clip(population, lb, ub)

            scores = np.array([func(ind) for ind in population])
            evaluations += self.population_size

            better_mask = scores < personal_best_scores
            personal_best_scores[better_mask] = scores[better_mask]
            personal_best_positions[better_mask] = population[better_mask]
            if min(scores) < global_best_score:
                global_best_score = min(scores)
                global_best_position = population[np.argmin(scores)]

            for i in range(self.population_size):
                perturbation = np.random.normal(0, 1, self.dim) * (self.temperature / 1000) * self.adaptive_perturbation
                candidate = global_best_position + perturbation
                candidate = np.clip(candidate, lb, ub)
                candidate_score = func(candidate)
                evaluations += 1
                if evaluations >= self.budget:
                    break

                if candidate_score < global_best_score or np.random.rand() < np.exp((global_best_score - candidate_score) / self.temperature):
                    global_best_position = candidate
                    global_best_score = candidate_score

            self.temperature *= self.cooling_rate
            self.inertia_weight = max(self.min_inertia_weight, self.inertia_weight * 0.98)  # Enhanced dynamic inertia adjustment

        return global_best_position