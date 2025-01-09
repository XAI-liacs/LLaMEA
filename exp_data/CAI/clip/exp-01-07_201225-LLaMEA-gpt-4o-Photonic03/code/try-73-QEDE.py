import numpy as np

class QEDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(20, 5 + dim)  # Changed to adaptive population size
        self.population = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, np.inf)
        self.best_solution = None
        self.best_fitness = np.inf

    def __call__(self, func):
        bounds = np.vstack((func.bounds.lb, func.bounds.ub)).T
        evaluations = 0

        def decode_position(position):
            return bounds[:, 0] + ((bounds[:, 1] - bounds[:, 0]) * position)

        def evaluate_position(decoded_position):
            nonlocal evaluations
            fitness = np.array([func(ind) for ind in decoded_position])
            evaluations += len(decoded_position)
            return fitness

        def quantum_mutation(base, best, r1, r2):  # Modified mutation strategy
            phi = np.random.rand(self.dim)
            perturbation = 0.05 * (1 - evaluations / self.budget)  # Changed perturbation factor
            scaling_factor = 0.8 * (1 - evaluations / self.budget)  # Adjusted scaling factor
            return base + phi * (best - base) + (1 - phi) * (r1 - r2) * scaling_factor + phi * (np.random.rand(self.dim) - 0.5) * perturbation

        def adaptive_de_crossover(target, mutant):  # New crossover method with adaptive strategy
            rand_vec = np.random.rand(self.dim)
            crossover_rate = 0.3 + 0.7 * (self.budget - evaluations) / self.budget  # Decrease crossover rate over time
            return np.where(rand_vec < crossover_rate, mutant, target)

        while evaluations < self.budget:
            decoded_population = decode_position(self.population)
            fitness = evaluate_position(decoded_population)

            for i in range(self.population_size):
                if fitness[i] < self.fitness[i]:
                    self.fitness[i] = fitness[i]
                    if fitness[i] < self.best_fitness:
                        self.best_fitness = fitness[i]
                        self.best_solution = self.population[i]

            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = quantum_mutation(a, self.best_solution, b, c)
                mutant = np.clip(mutant, 0, 1)
                trial = adaptive_de_crossover(self.population[i], mutant)  # Apply the new crossover

                trial_decoded = decode_position(trial)
                trial_fitness = func(trial_decoded)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial

        best_solution = decode_position(self.best_solution)
        return best_solution, self.best_fitness