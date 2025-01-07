import numpy as np

class QuantumSwarmDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.quantum_factor_initial = 0.4
        self.quantum_factor_final = 0.1
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability

    def quantum_update(self, position, global_best, eval_count):
        delta = np.random.rand(self.dim)
        lambda_factor = eval_count / self.budget
        quantum_factor = self.quantum_factor_initial * (1 - lambda_factor) + self.quantum_factor_final * lambda_factor
        new_position = position + quantum_factor * (global_best - position) * delta
        return new_position

    def differential_evolution(self, pop, bounds):
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.F * (b - c), bounds[:, 0], bounds[:, 1])
            crossover = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover, mutant, pop[i])
            pop[i] = trial
        return pop

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in pop])
        global_best = pop[np.argmin(fitness)]
        global_best_value = fitness.min()

        eval_count = self.population_size

        while eval_count < self.budget:
            pop = self.differential_evolution(pop, bounds)
            for i in range(self.population_size):
                trial = self.quantum_update(pop[i], global_best, eval_count)
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