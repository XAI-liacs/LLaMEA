import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30, c1=2.0, c2=2.0, w=0.7):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub

        # Initialize particle positions and velocities
        positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))

        # Initialize personal and global bests
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.num_particles, np.Inf)

        global_best_position = None
        global_best_score = np.Inf

        # Evaluation counter
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.num_particles):
                if evaluations >= self.budget:
                    break
                
                current_position = positions[i]
                score = func(current_position)
                evaluations += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = current_position

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = current_position

            # Update velocities and positions
            r1, r2 = np.random.rand(2)
            velocities = self.w * velocities + self.c1 * r1 * (personal_best_positions - positions) + self.c2 * r2 * (global_best_position - positions)
            positions = positions + velocities
            positions = np.clip(positions, lb, ub)

        # Store the best solution found
        self.f_opt = global_best_score
        self.x_opt = global_best_position

        return self.f_opt, self.x_opt