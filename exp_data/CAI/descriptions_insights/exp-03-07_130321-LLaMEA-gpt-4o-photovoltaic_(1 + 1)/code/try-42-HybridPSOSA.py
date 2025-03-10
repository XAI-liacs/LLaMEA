import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.inertia_weight = 0.7
        self.cognitive_coef = 1.5
        self.social_coef = 1.8
        self.momentum_coef = 0.1  # New line
        self.temperature = 1000
        self.cooling_rate = 0.96 
        self.min_inertia_weight = 0.4

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population_size = int(max(5, self.budget / (10 * self.dim)))
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)
        evaluations = self.population_size

        while evaluations < self.budget:
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive_term = self.cognitive_coef * (r1 ** 2) * (personal_best_positions - population)
            social_term = self.social_coef * (r2 ** 2) * (global_best_position - population)
            velocities = self.inertia_weight * velocities + cognitive_term + social_term + self.momentum_coef * velocities  # Modified line
            velocities *= np.random.normal(0.9, 0.1, velocities.shape)
            population = population + 0.5 * velocities
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
                perturbation_strength = np.exp(-evaluations / (0.1 * self.budget))
                perturbation = np.random.normal(0, perturbation_strength, self.dim)
                if np.random.rand() < 0.1:  # New condition
                    candidate = global_best_position + perturbation
                else:
                    candidate = global_best_position
                candidate = np.clip(candidate, lb, ub)
                candidate_score = func(candidate)
                evaluations += 1
                if evaluations >= self.budget:
                    break

                if candidate_score < global_best_score or np.random.rand() < np.exp((global_best_score - candidate_score) / self.temperature):
                    global_best_position = candidate
                    global_best_score = candidate_score

            self.temperature *= self.cooling_rate
            self.inertia_weight = max(self.min_inertia_weight, self.inertia_weight * 0.99 + 0.01 * (global_best_score / np.mean(personal_best_scores)))

        return global_best_position