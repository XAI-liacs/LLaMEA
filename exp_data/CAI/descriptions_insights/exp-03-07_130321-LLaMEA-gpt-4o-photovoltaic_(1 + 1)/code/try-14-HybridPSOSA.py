import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.inertia_weight = 0.7
        self.cognitive_coef = 1.5
        self.social_coef = 1.7  # Adjusted social coefficient for enhanced social learning
        self.temperature = 1000
        self.cooling_rate = 0.97  # Adjusted cooling rate for more gradual cooling
        self.min_inertia_weight = 0.4

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
            # Update velocities and positions
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive_term = self.cognitive_coef * r1 * (personal_best_positions - population)
            social_term = self.social_coef * r2 * (global_best_position - population)
            velocities = self.inertia_weight * velocities + cognitive_term + social_term
            population = population + 0.6 * velocities  # Adjusted position update for better exploration-exploitation balance
            population = np.clip(population, lb, ub)

            # Evaluate new solutions
            scores = np.array([func(ind) for ind in population])
            evaluations += self.population_size

            # Update personal best and global best
            better_mask = scores < personal_best_scores
            personal_best_scores[better_mask] = scores[better_mask]
            personal_best_positions[better_mask] = population[better_mask]
            if min(scores) < global_best_score:
                global_best_score = min(scores)
                global_best_position = population[np.argmin(scores)]

            # Perform SA-inspired local search around the global best
            for i in range(self.population_size):
                perturbation = np.random.normal(0, 1, self.dim) * (self.temperature / 1000)
                candidate = global_best_position + perturbation
                candidate = np.clip(candidate, lb, ub)
                candidate_score = func(candidate)
                evaluations += 1
                if evaluations >= self.budget:
                    break

                # SA acceptance criterion
                if candidate_score < global_best_score or np.random.rand() < np.exp((global_best_score - candidate_score) / self.temperature):
                    global_best_position = candidate
                    global_best_score = candidate_score

            # Cooling down
            self.temperature *= self.cooling_rate
            self.inertia_weight = max(self.min_inertia_weight, self.inertia_weight * 0.99)

        return global_best_position