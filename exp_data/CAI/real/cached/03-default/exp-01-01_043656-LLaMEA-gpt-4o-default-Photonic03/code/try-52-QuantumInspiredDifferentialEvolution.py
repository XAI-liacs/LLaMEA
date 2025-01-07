import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.quantum_factor_initial = 0.3
        self.quantum_factor_final = 0.1

    def quantum_perturbation(self, vector, best, eval_count):
        delta = np.random.rand(self.dim)
        lambda_factor = (eval_count / self.budget)
        quantum_factor = self.quantum_factor_initial * (1 - lambda_factor) + self.quantum_factor_final * lambda_factor
        perturbed_vector = vector + quantum_factor * (best - vector) * delta
        return perturbed_vector

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_vector = pop[best_idx]
        best_fitness = fitness[best_idx]

        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = pop[indices]
                mutant = x1 + self.mutation_factor * (x2 - x3)
                mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])

                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, pop[i])
                trial = self.quantum_perturbation(trial, best_vector, eval_count)
                trial = np.clip(trial, bounds[:, 0], bounds[:, 1])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_vector = trial
                        best_fitness = trial_fitness

                if eval_count >= self.budget:
                    break

        return best_vector