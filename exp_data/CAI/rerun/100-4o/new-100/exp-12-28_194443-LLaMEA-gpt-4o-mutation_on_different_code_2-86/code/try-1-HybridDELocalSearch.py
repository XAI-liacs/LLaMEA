import numpy as np

class HybridDELocalSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 10 + int(0.5 * dim)
        self.F = 0.9  # Increased Mutation factor
        self.CR = 0.9  # Crossover probability

    def differential_evolution(self, func, pop, func_evals):
        for _ in range(func_evals):
            for i in range(self.pop_size):
                if func_evals >= self.budget:
                    break

                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, pop[i])

                f = func(trial)
                if f < func(pop[i]):
                    pop[i] = trial
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial

                func_evals += 1

        return pop, func_evals

    def local_search(self, func, best):
        eps = 0.01
        for _ in range(20):  # Fixed small number of local steps
            if self.budget <= 0:
                break
            candidate = np.clip(best + eps * np.random.randn(self.dim), func.bounds.lb, func.bounds.ub)
            f = func(candidate)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = candidate
                best = candidate
            self.budget -= 1
        return best

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        func_evals = 0

        # Initial evaluation
        for i, individual in enumerate(pop):
            f = func(individual)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = individual
            func_evals += 1

        # Main loop of the algorithm
        while func_evals < self.budget:
            pop, func_evals = self.differential_evolution(func, pop, func_evals)
            best_idx = np.argmin([func(ind) for ind in pop])
            best_individual = pop[best_idx]
            best_individual = self.local_search(func, best_individual)

        return self.f_opt, self.x_opt