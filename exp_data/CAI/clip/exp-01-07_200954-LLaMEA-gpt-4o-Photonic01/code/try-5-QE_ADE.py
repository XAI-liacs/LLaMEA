import numpy as np

class QE_ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.F_min = 0.5
        self.F_max = 0.9
        self.CR = 0.9
        self.iterations = budget // self.population_size

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        global_best_idx = np.argmin(fitness)
        global_best_position = population[global_best_idx]
        global_best_fitness = fitness[global_best_idx]

        for iteration in range(self.iterations):
            if evals >= self.budget:
                break

            progress_ratio = iteration / self.iterations
            F = self.F_min + (self.F_max - self.F_min) * progress_ratio

            new_population = np.copy(population)
            for i in range(self.population_size):
                idxs = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[idxs]
                mutant_vector = x1 + F * (x2 - x3)
                mutant_vector = np.clip(mutant_vector, lb, ub)

                crossover_mask = np.random.rand(self.dim) < self.CR
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True

                trial_vector = np.where(crossover_mask, mutant_vector, population[i])
                trial_fitness = func(trial_vector)
                evals += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial_vector
                    fitness[i] = trial_fitness

                    if trial_fitness < global_best_fitness:
                        global_best_position = trial_vector
                        global_best_fitness = trial_fitness

            # Adaptive quantum-inspired perturbation
            sigma = self.F_max * (1 - progress_ratio)
            gaussian_noise = np.random.normal(0, sigma, (self.population_size, self.dim))
            quantum_population = np.clip(new_population + gaussian_noise, lb, ub)

            quantum_fitness = np.array([func(ind) for ind in quantum_population])
            evals += self.population_size

            improved_mask = quantum_fitness < fitness
            population[improved_mask] = quantum_population[improved_mask]
            fitness[improved_mask] = quantum_fitness[improved_mask]

            if np.min(fitness) < global_best_fitness:
                global_best_idx = np.argmin(fitness)
                global_best_position = population[global_best_idx]
                global_best_fitness = fitness[global_best_idx]

        return global_best_position, global_best_fitness