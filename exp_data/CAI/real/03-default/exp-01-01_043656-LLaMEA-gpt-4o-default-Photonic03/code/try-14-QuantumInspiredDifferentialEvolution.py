import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.mutation_factor_initial = 0.8
        self.mutation_factor_final = 0.5
        self.crossover_rate = 0.9
        self.quantum_factor_initial = 0.3
        self.quantum_factor_final = 0.1

    def quantum_mutation(self, target, donor, global_best, eval_count):
        delta = np.random.rand(self.dim)
        lambda_factor = eval_count / self.budget
        quantum_factor = self.quantum_factor_initial * (1 - lambda_factor) + self.quantum_factor_final * lambda_factor
        return donor + quantum_factor * (global_best - target) * delta

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in pop])
        global_best = pop[np.argmin(fitness)]
        global_best_value = fitness.min()

        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
                a, b, c = pop[indices]
                lambda_factor = eval_count / self.budget
                mutation_factor = self.mutation_factor_initial * (1 - lambda_factor) + self.mutation_factor_final * lambda_factor
                
                donor = np.clip(a + mutation_factor * (b - c), bounds[:, 0], bounds[:, 1])

                trial = np.array([donor[j] if np.random.rand() < self.crossover_rate else pop[i, j] for j in range(self.dim)])
                trial = np.clip(trial, bounds[:, 0], bounds[:, 1])

                trial_value = func(trial)
                eval_count += 1
                if trial_value < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_value
                    if trial_value < global_best_value:
                        global_best = trial
                        global_best_value = trial_value

                if eval_count >= self.budget:
                    break

        return global_best