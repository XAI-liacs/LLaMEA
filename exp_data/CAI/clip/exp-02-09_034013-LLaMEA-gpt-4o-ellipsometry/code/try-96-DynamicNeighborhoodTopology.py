import numpy as np

class DynamicNeighborhoodTopology:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.neighborhood_size = max(3, int(self.population_size / 3))
        self.inertia_weight = 0.7
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.initial_velocity = 0.5

    def self_adaptive_local_search(self, individual, lb, ub):
        perturbation = np.random.normal(scale=0.1, size=self.dim)
        new_position = individual + perturbation
        return np.clip(new_position, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-self.initial_velocity, self.initial_velocity, (self.population_size, self.dim))
        scores = np.full(self.population_size, np.inf)

        personal_best_positions = population.copy()
        personal_best_scores = np.full(self.population_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Evaluate current position
                current_score = func(population[i])
                evaluations += 1

                # Update personal best
                if current_score < personal_best_scores[i]:
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = population[i].copy()

                # Update global best
                if current_score < global_best_score:
                    global_best_score = current_score
                    global_best_position = population[i].copy()

            # Dynamic Neighborhood Topology
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Select neighbors randomly
                neighbors_indices = np.random.choice(self.population_size, self.neighborhood_size, replace=False)
                local_best_index = neighbors_indices[np.argmin(personal_best_scores[neighbors_indices])]

                # Calculate new velocity
                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    self.inertia_weight * velocities[i] +
                    self.cognitive_constant * r1 * (personal_best_positions[i] - population[i]) +
                    self.social_constant * r2 * (personal_best_positions[local_best_index] - population[i])
                )

                # Update position
                population[i] = np.clip(population[i] + velocities[i], lb, ub)

                # Self-adaptive local search
                if np.random.rand() < 0.05:  # Small probability
                    candidate = self.self_adaptive_local_search(population[i], lb, ub)
                    candidate_score = func(candidate)
                    evaluations += 1
                    if candidate_score < personal_best_scores[i]:
                        personal_best_scores[i] = candidate_score
                        personal_best_positions[i] = candidate
                        if candidate_score < global_best_score:
                            global_best_score = candidate_score
                            global_best_position = candidate.copy()

        return global_best_position, global_best_score