import numpy as np

class ImprovedHybridPSODEEfficient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.pso_params = {'w': 0.5, 'c1': 1.5, 'c2': 1.5}
        self.de_params = {'F': 0.5, 'CR': 0.9}

    def __call__(self, func):
        particles = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        pbest_particles = particles.copy()
        pbest_values = np.full(self.population_size, np.inf)
        gbest_value = np.inf

        for _ in range(self.budget):
            fitness_values = np.apply_along_axis(func, 1, particles)

            update_mask_pso = fitness_values < pbest_values
            pbest_values = np.where(update_mask_pso, fitness_values, pbest_values)
            pbest_particles = np.where(update_mask_pso[:, None], particles, pbest_particles)

            gbest_index = np.argmin(fitness_values)
            if fitness_values[gbest_index] < gbest_value:
                gbest_value = fitness_values[gbest_index]

            mutants = particles + self.de_params['F'] * (pbest_particles - particles) + self.de_params['F'] * (
                        particles[np.random.choice(self.population_size, size=self.population_size, replace=True)] - particles[
                    np.random.choice(self.population_size, size=self.population_size, replace=True)])
            trials = mutants.copy()

            cr_mask = np.random.rand(self.population_size, self.dim) > self.de_params['CR']
            trials = np.where(cr_mask[:, None], particles, trials)

            trial_fitness_values = np.apply_along_axis(func, 1, trials)
            update_mask_de = trial_fitness_values < pbest_values
            pbest_values = np.where(update_mask_de, trial_fitness_values, pbest_values)
            pbest_particles = np.where(update_mask_de[:, None], trials, pbest_particles)

            gbest_index_de = np.argmin(trial_fitness_values)
            if trial_fitness_values[gbest_index_de] < gbest_value:
                gbest_value = trial_fitness_values[gbest_index_de]

            particles = trials

        return pbest_particles[np.argmin(pbest_values)]