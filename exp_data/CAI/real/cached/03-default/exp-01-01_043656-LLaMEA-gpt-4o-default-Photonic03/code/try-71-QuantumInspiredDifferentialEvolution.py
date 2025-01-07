import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.quantum_factor_initial = 0.3
        self.quantum_factor_final = 0.1

    def quantum_perturbation(self, vector, best, eval_count):
        delta = np.random.rand(self.dim)
        lambda_factor = eval_count / self.budget  # Adaptive quantum factor
        quantum_factor = (self.quantum_factor_initial * (1 - lambda_factor) 
                          + self.quantum_factor_final * lambda_factor)
        perturbation = quantum_factor * (best - vector) * delta
        return vector + perturbation

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                # Mutation
                idxs = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = pop[idxs]
                mutant = x1 + self.F * (x2 - x3)
                mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Quantum perturbation
                trial = self.quantum_perturbation(trial, pop[np.argmin(fitness)], eval_count)
                trial = np.clip(trial, bounds[:, 0], bounds[:, 1])

                trial_fitness = func(trial)
                eval_count += 1

                # Selection
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                if eval_count >= self.budget:
                    break

        return pop[np.argmin(fitness)]