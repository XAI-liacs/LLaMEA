import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.particle_count = min(40, 2 * dim)
        self.w = 0.5  # inertia weight
        self.c1 = 2.0  # cognitive (particle) weight
        self.c2 = 2.0  # social (swarm) weight

    def __call__(self, func):
        # Initialize particles
        particles = np.random.uniform(self.lb, self.ub, (self.particle_count, self.dim))
        velocities = np.random.uniform(-1, 1, (self.particle_count, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.particle_count
        annealing_factor = 0.95

        while evaluations < self.budget:
            for i in range(self.particle_count):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (personal_best_positions[i] - particles[i])
                    + self.c2 * r2 * (global_best_position - particles[i])
                )
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.lb, self.ub)

                current_score = func(particles[i])
                evaluations += 1

                if current_score < personal_best_scores[i]:
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = particles[i].copy()

                if current_score < global_best_score:
                    global_best_score = current_score
                    global_best_position = particles[i].copy()

                if evaluations >= self.budget:
                    break

            # Simulated annealing on velocity to focus search
            self.w *= annealing_factor

        return global_best_position, global_best_score