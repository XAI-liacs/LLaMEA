import numpy as np

class HybridSADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        T0 = 1.0
        Tf = 1e-3
        lb, ub = func.bounds.lb, func.bounds.ub

        # Dynamic adaptation of crossover probability and differential weight
        CR_min, CR_max = 0.1, 0.9
        F_min, F_max = 0.5, 1.0

        pop_size = 20  # Increased population size for diversity
        population = np.random.uniform(lb, ub, size=(pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.update_optimum(population, fitness)

        for evals in range(self.budget - pop_size):
            CR = CR_min + (CR_max - CR_min) * (1 - evals / (self.budget - pop_size))
            F = F_min + (F_max - F_min) * np.random.rand()

            T = T0 * (Tf / T0) ** (evals / (self.budget - pop_size))

            idxs = np.random.choice(np.arange(pop_size), size=3, replace=False)
            x0, x1, x2 = population[idxs]

            mutant = x0 + F * (x1 - x2)
            mutant = np.clip(mutant, lb, ub)

            trial = np.where(np.random.rand(self.dim) < CR, mutant, x0)

            f_trial = func(trial)
            f_target = func(x0)
            delta = f_trial - f_target
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                population[idxs[0]] = trial
                fitness[idxs[0]] = f_trial
                self.update_optimum([trial], [f_trial])

        return self.f_opt, self.x_opt

    def update_optimum(self, candidates, fitness):
        for i, f in enumerate(fitness):
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = candidates[i]