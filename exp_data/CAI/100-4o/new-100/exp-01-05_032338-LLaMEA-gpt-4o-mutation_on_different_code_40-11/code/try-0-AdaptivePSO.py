import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.c1 = 2.0  # Cognitive weight
        self.c2 = 2.0  # Social weight
        self.w_max = 0.9  # Max inertia weight
        self.w_min = 0.4  # Min inertia weight

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        particles = np.random.uniform(bounds[0], bounds[1], (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))

        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.swarm_size, np.inf)

        global_best_score = np.inf
        global_best_position = np.zeros(self.dim)

        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Evaluate current position
                score = func(particles[i])
                evaluations += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i]

            # Update velocities and positions
            w = self.w_max - ((self.w_max - self.w_min) * evaluations / self.budget)
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_component = self.c1 * r1 * (personal_best_positions[i] - particles[i])
                social_component = self.c2 * r2 * (global_best_position - particles[i])

                velocities[i] = w * velocities[i] + cognitive_component + social_component
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], bounds[0], bounds[1])

            if evaluations >= self.budget:
                break

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt