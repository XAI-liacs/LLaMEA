import numpy as np

class HybridPSOLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

        # PSO parameters
        self.num_particles = 30
        self.inertia_weight = 0.9
        self.cognitive_param = 1.7
        self.social_param = 1.5
        
        # Local search parameters
        self.local_search_radius = 0.05
        self.local_search_steps = 15

    def __call__(self, func):
        num_evaluations = 0
        bounds = func.bounds
        lb = bounds.lb
        ub = bounds.ub

        # Initialize particles
        positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1.2, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([float('inf')] * self.num_particles)
        
        # Evaluate initial solutions
        for i in range(self.num_particles):
            score = func(positions[i])
            num_evaluations += 1
            personal_best_scores[i] = score
            if num_evaluations >= self.budget:
                return positions[i]
        
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        # Main loop
        stagnation_count = 0
        while num_evaluations < self.budget:
            for i in range(self.num_particles):
                neighborhood_size = np.random.randint(3, 7)
                neighborhood = np.random.choice(self.num_particles, neighborhood_size, replace=False)
                neighborhood_best = personal_best_positions[neighborhood[np.argmin(personal_best_scores[neighborhood])]]
                
                r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
                self.inertia_weight *= 0.98
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_param * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social_param * r2 * (global_best_position - positions[i]) +
                                 0.6 * r3 * (neighborhood_best - positions[i]))
                velocities[i] *= 0.7
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

                score = func(positions[i])
                num_evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]
                    stagnation_count = 0

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]
                    stagnation_count = 0

                if num_evaluations >= self.budget:
                    return global_best_position

            stagnation_count += 1
            if stagnation_count > 15:
                restart_indices = np.random.choice(self.num_particles, 5, replace=False)
                positions[restart_indices] = np.random.uniform(lb, ub, (5, self.dim))
                stagnation_count = 0

            # Local Search with adaptive radius
            new_best_position = self.local_search(global_best_position, func, lb, ub, num_evaluations, stagnation_count)
            if new_best_position is not None:
                global_best_position = new_best_position

        return global_best_position

    def local_search(self, position, func, lb, ub, num_evaluations, stagnation_count):  # Changed line
        best_position = position
        best_score = func(best_position)
        num_evaluations += 1

        if num_evaluations >= self.budget:
            return None

        cooling_rate = 0.95 - stagnation_count * 0.005  # New: Adjust cooling rate based on stagnation_count
        local_radius = self.local_search_radius
        no_improvement_steps = 0

        for _ in range(self.local_search_steps):
            local_radius *= cooling_rate
            candidate_position = best_position + np.random.uniform(-local_radius, local_radius, self.dim)
            candidate_position = np.clip(candidate_position, lb, ub)
            candidate_score = func(candidate_position)
            num_evaluations += 1

            if candidate_score < best_score:
                best_position, best_score = candidate_position, candidate_score
                no_improvement_steps = 0

                if num_evaluations >= self.budget:
                    return None
            else:
                no_improvement_steps += 1

            if no_improvement_steps > 5:
                break

        return best_position