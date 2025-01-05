import numpy as np

class BioInspiredWaveOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.amplitude = 0.5
        self.frequency = 0.1
        self.damping_factor = 0.99

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        wave_population = np.random.rand(self.population_size, self.dim)
        position_population = self.wave_to_position(wave_population, lb, ub)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = position_population[best_index]

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Update wave parameters
                self.amplitude *= self.damping_factor
                wave_population[i] = self.update_wave_parameters(wave_population[i], wave_population[best_index])

                # Convert wave representation to classical position
                position_population[i] = self.wave_to_position(wave_population[i], lb, ub)

                # Evaluate new position
                new_fitness = func(position_population[i])
                evaluations += 1

                # Selection: keep the better solution
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness

                # Update best position
                if new_fitness < fitness[best_index]:
                    best_index = i
                    best_position = position_population[i]

                if evaluations >= self.budget:
                    break

        return best_position, fitness[best_index]

    def wave_to_position(self, wave_parameters, lb, ub):
        # Convert wave parameters to classical positions in the search space
        position = lb + wave_parameters * (ub - lb)
        return position

    def update_wave_parameters(self, wave_parameters, best_wave_parameters):
        # Wave behavior emulation for updates
        delta_wave = self.amplitude * np.sin(self.frequency * (best_wave_parameters - wave_parameters))
        new_wave_parameters = wave_parameters + delta_wave
        new_wave_parameters = np.clip(new_wave_parameters, 0, 1)
        return new_wave_parameters