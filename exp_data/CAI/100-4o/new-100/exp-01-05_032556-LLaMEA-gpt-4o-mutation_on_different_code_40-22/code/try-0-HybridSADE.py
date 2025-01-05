import numpy as np

class HybridSADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Simulated Annealing with Adaptive Differential Evolution
        T0 = 1.0  # Initial temperature for SA
        Tf = 1e-3  # Final temperature
        CR = 0.9  # Crossover probability for DE
        F = 0.8  # Differential weight for DE
        lb, ub = func.bounds.lb, func.bounds.ub

        # Initialize population
        pop_size = 10
        population = np.random.uniform(lb, ub, size=(pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.update_optimum(population, fitness)

        for evals in range(self.budget - pop_size):
            T = T0 * (Tf/T0) ** (evals / (self.budget - pop_size))

            # Select a target vector
            idxs = np.random.choice(np.arange(pop_size), size=3, replace=False)
            x0, x1, x2 = population[idxs]

            # Generate mutant vector
            mutant = x0 + F * (x1 - x2)
            mutant = np.clip(mutant, lb, ub)

            # Crossover
            trial = np.where(np.random.rand(self.dim) < CR, mutant, x0)

            # Simulated Annealing Acceptance Criterion
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