import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=50, inertia=0.7, cognitive=1.5, social=1.5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize particle positions and velocities
        positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, np.Inf)
        global_best_position = None
        global_best_score = np.Inf

        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.num_particles):
                # Evaluate the current position
                score = func(positions[i])
                eval_count += 1

                # Update personal and global bests
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

                # If budget is exhausted, break
                if eval_count >= self.budget:
                    break

            # Update velocities and positions
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            for i in range(self.num_particles):
                velocities[i] = (self.inertia * velocities[i] +
                                self.cognitive * r1 * (personal_best_positions[i] - positions[i]) +
                                self.social * r2 * (global_best_position - positions[i]))
                positions[i] = np.clip(positions[i] + velocities[i], func.bounds.lb, func.bounds.ub)

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt