import numpy as np

class QuantumInspiredAdaptiveOBLDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F_min, self.F_max = 0.5, 0.9
        self.CR_min, self.CR_max = 0.7, 0.9
        self.history = []

    def quantum_position(self, lb, ub):
        quantum_particles = np.random.uniform(0, 1, (self.population_size, self.dim))
        return lb + (ub - lb) * np.cos(np.pi * quantum_particles)

    def opposition_based_learning(self, x, lb, ub):
        return lb + ub - x

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = self.quantum_position(lb, ub)
        fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(fitness)
        best_global = pop[best_idx]
        evaluations = self.population_size

        while evaluations < self.budget:
            next_pop = np.zeros((self.population_size, self.dim))

            for i in range(self.population_size):
                indices = np.random.choice(range(self.population_size), 3, replace=False)
                x0, x1, x2 = pop[indices]
                F = np.random.uniform(self.F_min, self.F_max)
                CR = np.random.uniform(self.CR_min, self.CR_max)

                mutant = x0 + F * (x1 - x2)
                mutant = np.clip(mutant, lb, ub)

                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, pop[i])
                trial_obl = self.opposition_based_learning(trial, lb, ub)
                trial_fitness = func(trial)
                trial_obl_fitness = func(trial_obl)
                evaluations += 2

                if trial_fitness < fitness[i] or trial_obl_fitness < fitness[i]:
                    if trial_fitness < trial_obl_fitness:
                        next_pop[i] = trial
                        fitness[i] = trial_fitness
                    else:
                        next_pop[i] = trial_obl
                        fitness[i] = trial_obl_fitness

                    if fitness[i] < fitness[best_idx]:
                        best_idx = i
                        best_global = next_pop[i]
                else:
                    next_pop[i] = pop[i]

            pop = next_pop
            self.history.append(best_global)

        return best_global