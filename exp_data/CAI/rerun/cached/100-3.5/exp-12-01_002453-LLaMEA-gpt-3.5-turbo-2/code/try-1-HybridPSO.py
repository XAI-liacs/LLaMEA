import numpy as np

class HybridPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.max_iter = budget // self.num_particles
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def local_search(current_position, current_value):
            epsilon = 0.01
            for _ in range(10):
                new_position = current_position + epsilon * np.random.uniform(-1, 1, size=self.dim)
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                new_value = func(new_position)
                if new_value < current_value:
                    current_position = new_position
                    current_value = new_value
            return current_position, current_value

        best_global_position = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
        best_global_value = func(best_global_position)

        for _ in range(self.max_iter):
            for _ in range(self.num_particles):
                particle_position = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
                particle_value = func(particle_position)

                if particle_value < best_global_value:
                    best_global_position = particle_position
                    best_global_value = particle_value

                particle_position, particle_value = local_search(particle_position, particle_value)

        return best_global_position