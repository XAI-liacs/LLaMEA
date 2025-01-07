import numpy as np

class HybridCuckooPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n_nests = 25
        beta = 3 / 2 
        pa = 0.2 + 0.05 * np.random.rand()
        n_particles = 30
        w = 0.5
        c1 = 1.5 
        c2 = 1.5 + 0.5 * np.random.rand()
        bounds = np.array([func.bounds.lb, func.bounds.ub])

        nests = np.random.uniform(bounds[0], bounds[1], (n_nests, self.dim))
        particles = np.random.uniform(bounds[0], bounds[1], (n_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (n_particles, self.dim))

        nest_fitness = np.apply_along_axis(func, 1, nests)
        personal_best_positions = np.copy(particles)
        personal_best_fitness = np.apply_along_axis(func, 1, particles)

        global_best_idx = np.argmin(nest_fitness)
        global_best_position = nests[global_best_idx]
        global_best_fitness = nest_fitness[global_best_idx]

        eval_count = 0

        while eval_count < self.budget:
            explore_factor = 1 - eval_count / self.budget
            new_nests = self.hybrid_levy_flights(nests, global_best_position, bounds, beta, explore_factor)
            new_fitness = np.apply_along_axis(func, 1, new_nests)
            eval_count += n_nests

            replace = new_fitness < nest_fitness
            nests[replace] = new_nests[replace]
            nest_fitness[replace] = new_fitness[replace]

            abandon_indices = np.random.rand(n_nests) < pa
            nests[abandon_indices] = np.random.uniform(bounds[0], bounds[1], (np.sum(abandon_indices), self.dim))
            nest_fitness[abandon_indices] = np.apply_along_axis(func, 1, nests[abandon_indices])
            eval_count += np.sum(abandon_indices)

            current_global_best_idx = np.argmin(nest_fitness)
            if nest_fitness[current_global_best_idx] < global_best_fitness:
                global_best_position = nests[current_global_best_idx]
                global_best_fitness = nest_fitness[current_global_best_idx]

            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            w = 0.9 - 0.5 * (eval_count / self.budget)
            c1_adaptive = c1 * explore_factor
            c2_adaptive = c2 * (1 - explore_factor)
            velocities = (w * velocities +
                          c1_adaptive * r1 * (personal_best_positions - particles) +
                          c2_adaptive * r2 * (global_best_position - particles))
            particles += velocities
            particles = np.clip(particles, bounds[0], bounds[1])

            particle_fitness = np.apply_along_axis(func, 1, particles)
            eval_count += n_particles

            better_fitness = particle_fitness < personal_best_fitness
            personal_best_positions[better_fitness] = particles[better_fitness]
            personal_best_fitness[better_fitness] = particle_fitness[better_fitness]

            current_particle_best_idx = np.argmin(personal_best_fitness)
            if personal_best_fitness[current_particle_best_idx] < global_best_fitness:
                global_best_position = personal_best_positions[current_particle_best_idx]
                global_best_fitness = personal_best_fitness[current_particle_best_idx]

        return global_best_position

    def hybrid_levy_flights(self, nests, best, bounds, beta, explore_factor):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        
        u = np.random.normal(0, sigma, nests.shape)
        v = np.random.normal(0, 1, nests.shape)
        step = u / np.abs(v) ** (1 / beta)

        step_size = 0.01 * step * (nests - best)
        new_nests = nests + explore_factor * step_size
        new_nests = np.clip(new_nests, bounds[0], bounds[1])
        return new_nests