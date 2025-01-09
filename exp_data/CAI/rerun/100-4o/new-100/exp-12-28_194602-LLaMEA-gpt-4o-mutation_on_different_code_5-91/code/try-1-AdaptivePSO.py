import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.f_opt = np.Inf
        self.x_opt = None
        self.swarm = None

    def initialize_swarm(self, bounds):
        self.swarm = []
        for _ in range(self.num_particles):
            position = np.random.uniform(bounds.lb, bounds.ub, self.dim)
            velocity = np.random.uniform(-1, 1, self.dim)
            self.swarm.append({'position': position, 'velocity': velocity,
                               'best_position': position, 'best_value': np.Inf})

    def update_velocity(self, particle, global_best_position, inertia_weight, c1, c2):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        cognitive = c1 * r1 * (particle['best_position'] - particle['position'])
        social = c2 * r2 * (global_best_position - particle['position'])
        particle['velocity'] = inertia_weight * particle['velocity'] + cognitive + social

    def update_position(self, particle, bounds):
        particle['position'] += particle['velocity']
        particle['position'] = np.clip(particle['position'], bounds.lb, bounds.ub)

    def __call__(self, func):
        self.initialize_swarm(func.bounds)
        global_best_position, global_best_value = None, np.Inf
        evals = 0

        while evals < self.budget:
            inertia_weight = 0.9 - 0.5 * (evals / self.budget) ** 2  # Changed to non-linear update
            c1 = c2 = 2.0 - 1.5 * (1 - evals / self.budget)

            for particle in self.swarm:
                current_value = func(particle['position'])
                evals += 1

                if current_value < particle['best_value']:
                    particle['best_value'] = current_value
                    particle['best_position'] = particle['position']

                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = particle['position']

                if evals >= self.budget:
                    break

            for particle in self.swarm:
                self.update_velocity(particle, global_best_position, inertia_weight, c1, c2)
                self.update_position(particle, func.bounds)

        self.f_opt, self.x_opt = global_best_value, global_best_position
        return self.f_opt, self.x_opt