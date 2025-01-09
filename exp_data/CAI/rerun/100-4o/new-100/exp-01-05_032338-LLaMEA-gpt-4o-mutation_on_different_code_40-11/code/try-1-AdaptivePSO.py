import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.c1_initial = 2.5  # Initial cognitive weight
        self.c2_initial = 1.5  # Initial social weight
        self.c1_final = 1.5    # Final cognitive weight
        self.c2_final = 2.5    # Final social weight
        self.w_max = 0.9  # Max inertia weight
        self.w_min = 0.4  # Min inertia weight

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        particles = np.random.uniform(bounds[0], bounds[1], (self.swarm_size, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.swarm_size, self.dim))

        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.swarm_size, np.inf)

        global_best_score = np.inf
        global_best_position = np.zeros(self.dim)

        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                score = func(particles[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i]

            w = self.w_max - ((self.w_max - self.w_min) * evaluations / self.budget)
            c1 = self.c1_initial - ((self.c1_initial - self.c1_final) * evaluations / self.budget)
            c2 = self.c2_initial + ((self.c2_final - self.c2_initial) * evaluations / self.budget)

            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_component = c1 * r1 * (personal_best_positions[i] - particles[i])
                social_component = c2 * r2 * (global_best_position - particles[i])

                velocities[i] = 0.7 * velocities[i] + cognitive_component + social_component
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], bounds[0], bounds[1])

            if evaluations >= self.budget:
                break

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt