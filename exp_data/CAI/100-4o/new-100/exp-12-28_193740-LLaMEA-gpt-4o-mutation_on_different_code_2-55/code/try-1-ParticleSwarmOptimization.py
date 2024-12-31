import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.inertia_weight_initial = 0.9  # Initial inertia weight
        self.inertia_weight_final = 0.4    # Final inertia weight
        self.cognitive_const = 1.5
        self.social_const = 1.5
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize particles' positions and velocities
        positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, np.Inf)
        global_best_position = None
        global_best_score = np.Inf
        
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.num_particles):
                if evaluations >= self.budget:
                    break
                # Evaluate particle
                score = func(positions[i])
                evaluations += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

            # Update inertia weight dynamically
            inertia_weight = self.inertia_weight_initial - ((self.inertia_weight_initial - self.inertia_weight_final) * (evaluations / self.budget))

            # Update velocities and positions
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            velocities = (inertia_weight * velocities 
                          + self.cognitive_const * r1 * (personal_best_positions - positions)
                          + self.social_const * r2 * (global_best_position - positions))
            positions = positions + velocities
            positions = np.clip(positions, func.bounds.lb, func.bounds.ub)  # Ensure within bounds

        self.f_opt = global_best_score
        self.x_opt = global_best_position

        return self.f_opt, self.x_opt