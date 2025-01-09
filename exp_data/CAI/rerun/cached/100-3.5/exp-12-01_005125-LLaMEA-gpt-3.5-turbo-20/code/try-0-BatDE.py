import numpy as np

class BatDE:
    def __init__(self, budget, dim, population_size=20, loudness=0.5, pulse_rate=0.5, f=0.5, cr=0.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.loudness = loudness
        self.pulse_rate = pulse_rate
        self.f = f
        self.cr = cr

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, size=self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget):
            population = [np.random.uniform(-5.0, 5.0, size=self.dim) for _ in range(self.population_size)]
            velocities = np.zeros((self.population_size, self.dim))

            for i, bat in enumerate(population):
                if np.random.rand() > self.pulse_rate:
                    frequencies = np.clip(np.random.normal(0.5, 0.1, self.dim), 0, 1)
                    velocities[i] += (bat - best_solution) * frequencies

                new_solution = bat + velocities[i]
                if np.random.rand() < self.loudness and np.all(new_solution >= -5.0) and np.all(new_solution <= 5.0):
                    trial_solution = bat + self.f * (best_solution - bat) + self.cr * (new_solution - population[np.random.randint(self.population_size)])
                    trial_fitness = func(trial_solution)
                    if trial_fitness < best_fitness:
                        best_solution = trial_solution
                        best_fitness = trial_fitness

            self.loudness *= 0.9
            self.pulse_rate *= 0.9

        return best_solution