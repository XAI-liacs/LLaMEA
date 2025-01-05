import numpy as np

class BioInspiredCooperativeBatAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, 5 * dim)
        self.frequency_min = 0.0
        self.frequency_max = 2.0
        self.alpha = 0.9  # loudness coefficient
        self.gamma = 0.9  # pulse rate coefficient
        self.loudness = 1.0  # initial loudness
        self.pulse_rate = 0.5  # initial pulse rate

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        velocities = np.zeros_like(pop)
        fitness = np.array([func(ind) for ind in pop])
        best_index = np.argmin(fitness)
        global_best = pop[best_index]
        global_best_value = fitness[best_index]

        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                frequency = self.frequency_min + (self.frequency_max - self.frequency_min) * np.random.rand()
                velocities[i] += (pop[i] - global_best) * frequency
                trial = pop[i] + velocities[i]
                trial = np.clip(trial, bounds[:, 0], bounds[:, 1])

                if np.random.rand() > self.pulse_rate:
                    # Exploitation phase: move globally towards the best known solution
                    trial = global_best + 0.001 * np.random.randn(self.dim)

                trial_value = func(trial)
                eval_count += 1
                if (trial_value < fitness[i]) and (np.random.rand() < self.loudness):
                    pop[i] = trial
                    fitness[i] = trial_value

                    if trial_value < global_best_value:
                        global_best = trial
                        global_best_value = trial_value

                # Update pulse rate and loudness
                self.pulse_rate = self.pulse_rate * (1 - np.exp(-self.gamma * eval_count / self.budget))
                self.loudness *= self.alpha

                if eval_count >= self.budget:
                    break

        return global_best