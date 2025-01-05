import numpy as np

class QuantumEnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.initial_quantum_factor = 0.3
        self.final_quantum_factor = 0.1
        self.f = 0.5    # Differential evolution control parameter
        self.cr = 0.9   # Crossover probability

    def quantum_update(self, target, mutant, eval_count):
        lambda_factor = eval_count / self.budget
        quantum_factor = self.initial_quantum_factor * (1 - lambda_factor) + self.final_quantum_factor * lambda_factor
        delta = np.random.rand(self.dim)
        new_position = target + quantum_factor * (mutant - target) * delta
        return new_position

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size
        best_idx = np.argmin(fitness)
        best = pop[best_idx]

        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = pop[indices]
                mutant = np.clip(x0 + self.f * (x1 - x2), bounds[:, 0], bounds[:, 1])

                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, pop[i])
                trial = self.quantum_update(trial, pop[i], eval_count)

                trial = np.clip(trial, bounds[:, 0], bounds[:, 1])
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best = trial

                if eval_count >= self.budget:
                    break

        return best