import numpy as np

class HybridPSOLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

        # PSO parameters
        self.num_particles = 30
        self.inertia_weight = 0.9  # Adjusted initial inertia weight
        self.cognitive_param = 1.5
        self.social_param = 1.5
        
        # Local search parameters
        self.local_search_radius = 0.05  # Reduced search radius
        self.local_search_steps = 15  # Increased search steps

    def __call__(self, func):
        num_evaluations = 0
        bounds = func.bounds
        lb = bounds.lb
        ub = bounds.ub

        # Initialize particles
        positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
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
        early_convergence_threshold = 0.001  # Added early convergence detection threshold
        while num_evaluations < self.budget:
            for i in range(self.num_particles):
                # Update velocity and position
                r1, r2 = np.random.rand(), np.random.rand()
                self.inertia_weight *= 0.99  # Adaptive inertia weight
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_param * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social_param * r2 * (global_best_position - positions[i]))
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

                # Evaluate and update personal best
                score = func(positions[i])
                num_evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

                if num_evaluations >= self.budget:
                    return global_best_position

            # Early convergence detection
            if global_best_score < early_convergence_threshold:
                return global_best_position

            # Local Search on the best solution found so far
            new_best_position = self.local_search(global_best_position, func, lb, ub, num_evaluations)
            if new_best_position is not None:
                global_best_position = new_best_position

        return global_best_position

    def local_search(self, position, func, lb, ub, num_evaluations):
        best_position = position
        best_score = func(best_position)
        num_evaluations += 1

        if num_evaluations >= self.budget:
            return None

        for _ in range(self.local_search_steps):
            candidate_position = best_position + np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dim)
            candidate_position = np.clip(candidate_position, lb, ub)
            candidate_score = func(candidate_position)
            num_evaluations += 1

            if candidate_score < best_score:
                best_position, best_score = candidate_position, candidate_score

                if num_evaluations >= self.budget:
                    return None

        return best_position