import numpy as np

class EMO_DGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.grid_resolution = 20
        self.grid = np.linspace(self.lower_bound, self.upper_bound, self.grid_resolution)

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        evaluations = 0

        while evaluations < self.budget:
            grid_counts = np.zeros((self.grid_resolution, self.dim))
            for ind in population:
                grid_indices = np.floor((ind - self.lower_bound) / ((self.upper_bound - self.lower_bound) / self.grid_resolution)).astype(int)
                grid_indices = np.clip(grid_indices, 0, self.grid_resolution - 1)
                grid_counts[tuple(grid_indices)] += 1

            grid_densities = np.sum(grid_counts, axis=1)
            new_population = []
            for _ in range(self.budget):
                selected_grid = np.random.choice(self.grid_resolution, p=grid_densities / np.sum(grid_densities))
                selected_inds = np.where(np.all(np.floor((population - self.lower_bound) / ((self.upper_bound - self.lower_bound) / self.grid_resolution)).astype(int) == selected_grid, axis=1))[0]
                selected_ind = population[np.random.choice(selected_inds)]
                new_population.append(selected_ind + np.random.normal(0, 0.1, self.dim))

            population = np.array(new_population)
            evaluations += len(new_population)

        return population[np.argmin([func(ind) for ind in population])]