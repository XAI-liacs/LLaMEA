import numpy as np

class HybridQuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.quantum_factor = 0.2
        self.F = 0.5  # Differential weight factor
        self.CR = 0.9  # Crossover probability

    def quantum_update(self, position, global_best, eval_count):
        delta = np.random.rand(self.dim)
        lambda_factor = (eval_count / self.budget)
        quantum_influence = self.quantum_factor * (1 - lambda_factor)
        new_position = position + quantum_influence * (global_best - position) * delta
        return new_position

    def differential_mutation(self, pop, idx, bounds):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = pop[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), bounds[:, 0], bounds[:, 1])
        return mutant

    def differential_crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in pop])
        global_best = pop[np.argmin(fitness)]
        global_best_value = fitness.min()

        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                mutant = self.differential_mutation(pop, i, bounds)
                trial = self.differential_crossover(pop[i], mutant)
                trial = self.quantum_update(trial, global_best, eval_count)
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