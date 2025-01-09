import numpy as np

class MemeticDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 20
        self.CR = 0.9  # Crossover probability
        self.F = 0.5   # Differential weight

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.pop_size

        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.F * (b - c), lb, ub)

                # Crossover
                trial = np.array([mutant[j] if np.random.rand() < self.CR else population[i, j] for j in range(self.dim)])

                # Local Search
                trial = self.local_search(trial, func)

                # Selection
                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

        return self.f_opt, self.x_opt

    def local_search(self, x, func):
        step_size = 0.1
        best_x = np.copy(x)
        best_f = func(x)
        for _ in range(10):
            for j in range(self.dim):
                perturbation = np.zeros(self.dim)
                perturbation[j] = step_size
                candidates = [x + perturbation, x - perturbation]
                for candidate in candidates:
                    candidate = np.clip(candidate, func.bounds.lb, func.bounds.ub)
                    f_candidate = func(candidate)
                    if f_candidate < best_f:
                        best_f = f_candidate
                        best_x = candidate
        return best_x