import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, population_size=30, inertia=0.7, cognitive=1.5, social=1.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize particles
        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.inf)

        for _ in range(self.budget // self.population_size):
            for i in range(self.population_size):
                score = func(particles[i])
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                if score < self.f_opt:
                    self.f_opt = score
                    self.x_opt = particles[i]

            global_best_position = personal_best_positions[np.argmin(personal_best_scores)]

            for i in range(self.population_size):
                # Update velocities and positions
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia * velocities[i] +
                                 self.cognitive * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.social * r2 * (global_best_position - particles[i]))
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

        return self.f_opt, self.x_opt