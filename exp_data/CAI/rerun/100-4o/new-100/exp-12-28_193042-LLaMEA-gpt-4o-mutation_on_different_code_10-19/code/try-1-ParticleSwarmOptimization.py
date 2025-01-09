import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.f_opt = np.Inf
        self.x_opt = None
        self.neighborhood_radius = 0.1  # New parameter for neighborhood search

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(low=lb, high=ub, size=(self.num_particles, self.dim))
        velocities = np.random.uniform(low=-abs(ub-lb), high=abs(ub-lb), size=(self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(self.num_particles, np.inf)
        
        global_best_position = None
        global_best_value = np.inf

        w_max, w_min = 0.9, 0.4
        c1, c2 = 2.05, 2.05
        k = 2 / abs(2 - c1 - c2 - np.sqrt((c1 + c2)**2 - 4*(c1 + c2)))

        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.num_particles):
                f_value = func(particles[i])
                evaluations += 1

                if f_value < personal_best_values[i]:
                    personal_best_values[i] = f_value
                    personal_best_positions[i] = particles[i]
                    # Local search within neighborhood
                    local_search_position = particles[i] + np.random.uniform(-self.neighborhood_radius, self.neighborhood_radius, self.dim)
                    local_search_position = np.clip(local_search_position, lb, ub)
                    local_search_value = func(local_search_position)
                    evaluations += 1
                    if local_search_value < personal_best_values[i]:
                        personal_best_values[i] = local_search_value
                        personal_best_positions[i] = local_search_position

                if f_value < global_best_value:
                    global_best_value = f_value
                    global_best_position = particles[i]

                if evaluations >= self.budget:
                    break

            inertia_weight = w_max - ((w_max - w_min) * (evaluations / self.budget))

            for i in range(self.num_particles):
                velocities[i] = (inertia_weight * velocities[i]
                                 + c1 * np.random.rand() * (personal_best_positions[i] - particles[i])
                                 + c2 * np.random.rand() * (global_best_position - particles[i]))
                velocities[i] *= k
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

        self.f_opt, self.x_opt = global_best_value, global_best_position
        return self.f_opt, self.x_opt