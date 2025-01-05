import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.mutation_factor = 0.5
        self.crossover_prob = 0.7
        self.quantum_factor_initial = 0.3
        self.quantum_factor_final = 0.1

    def quantum_interference(self, position, eval_count):
        quantum_factor = self.quantum_factor_initial * (1 - eval_count / self.budget) + self.quantum_factor_final * (eval_count / self.budget)
        interference_pattern = np.sin(np.pi * position + np.random.rand(self.dim)) * quantum_factor
        return interference_pattern

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = pop[indices]
                mutant = x0 + self.mutation_factor * (x1 - x2)
                mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])

                crossover = np.random.rand(self.dim) < self.crossover_prob
                trial = np.where(crossover, mutant, pop[i])

                quantum_adjustment = self.quantum_interference(trial, eval_count)
                trial = np.clip(trial + quantum_adjustment, bounds[:, 0], bounds[:, 1])

                trial_value = func(trial)
                eval_count += 1

                if trial_value < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_value

                if eval_count >= self.budget:
                    break

        best_index = np.argmin(fitness)
        return pop[best_index]