import numpy as np

class AdaptiveParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.inertia_weight = 0.5
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.local_search_intensity = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = lb + np.random.rand(self.population_size, self.dim) * (ub - lb)
        velocities = np.random.rand(self.population_size, self.dim) * (ub - lb) * 0.1
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(ind) for ind in positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coefficient * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social_coefficient * r2 * (global_best_position - positions[i]))
                positions[i] = positions[i] + velocities[i]

                # Apply boundary constraints
                positions[i] = np.clip(positions[i], lb, ub)

                # Evaluate new position
                new_score = func(positions[i])
                evaluations += 1

                # Check if current position is a new personal best
                if new_score < personal_best_scores[i]:
                    personal_best_scores[i] = new_score
                    personal_best_positions[i] = positions[i]

                # Check if current position is a new global best
                if new_score < personal_best_scores[global_best_index]:
                    global_best_index = i
                    global_best_position = positions[i]

                if evaluations >= self.budget:
                    break

            # Perform local search intensification
            if np.random.rand() < self.local_search_intensity:
                local_best_index = np.random.randint(self.population_size)
                local_best_position = personal_best_positions[local_best_index]
                local_search_radius = (ub - lb) * 0.05
                for j in range(self.dim):
                    perturbed_position = local_best_position + np.random.uniform(-local_search_radius[j], local_search_radius[j])
                    perturbed_position = np.clip(perturbed_position, lb[j], ub[j])
                    perturbed_score = func(perturbed_position)
                    evaluations += 1
                    
                    if perturbed_score < personal_best_scores[local_best_index]:
                        personal_best_scores[local_best_index] = perturbed_score
                        personal_best_positions[local_best_index] = perturbed_position

                    if perturbed_score < personal_best_scores[global_best_index]:
                        global_best_index = local_best_index
                        global_best_position = perturbed_position

                    if evaluations >= self.budget:
                        break

        return global_best_position, personal_best_scores[global_best_index]