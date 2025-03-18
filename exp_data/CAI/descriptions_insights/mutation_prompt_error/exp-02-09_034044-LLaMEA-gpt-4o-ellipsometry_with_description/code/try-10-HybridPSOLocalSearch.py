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
        self.qpso_beta = 0.5  # New parameter for QPSO

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
        while num_evaluations < self.budget:
            for i in range(self.num_particles):
                # Adaptive neighborhood search
                neighborhood = np.random.choice(self.num_particles, 5, replace=False)
                neighborhood_best = personal_best_positions[neighborhood[np.argmin(personal_best_scores[neighborhood])]]
                
                # QPSO updates
                self.inertia_weight *= 0.99
                mbest = np.mean(personal_best_positions, axis=0)
                phi = np.random.rand(self.dim)
                p = self.qpso_beta * global_best_position + (1 - self.qpso_beta) * mbest
                positions[i] = np.clip(p + phi * np.abs(global_best_position - positions[i]) * np.log(1/np.random.rand(self.dim)), lb, ub)

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