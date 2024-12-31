import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, swarm_size=30, c1=2.05, c2=2.05, w=0.729):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.c1 = c1  # cognitive coefficient
        self.c2 = c2  # social coefficient
        self.w = w    # inertia weight
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize particles
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.swarm_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(x) for x in personal_best_positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * r2 * (global_best_position - particles[i]))
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)
                score = func(particles[i])
                evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i]

                if evaluations >= self.budget:
                    break

        self.f_opt = global_best_score
        self.x_opt = global_best_position

        return self.f_opt, self.x_opt