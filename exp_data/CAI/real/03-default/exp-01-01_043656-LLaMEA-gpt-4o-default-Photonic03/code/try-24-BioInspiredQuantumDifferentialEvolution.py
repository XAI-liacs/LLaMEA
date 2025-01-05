import numpy as np

class BioInspiredQuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.quantum_factor_initial = 0.4
        self.quantum_factor_final = 0.1

    def quantum_mutation(self, target, donor, global_best, eval_count):
        delta = np.random.rand(self.dim)
        lambda_factor = (eval_count / self.budget)  # Adaptive quantum factor
        quantum_factor = self.quantum_factor_initial * (1 - lambda_factor) + self.quantum_factor_final * lambda_factor
        quantum_donor = donor + quantum_factor * (global_best - target) * delta
        return quantum_donor

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in pop])
        global_best = pop[np.argmin(fitness)]
        global_best_value = fitness.min()

        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = pop[indices[0]], pop[indices[1]], pop[indices[2]]
                mutant = x1 + self.F * (x2 - x3)
                mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])

                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pop[i])
                trial = self.quantum_mutation(pop[i], trial, global_best, eval_count)
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