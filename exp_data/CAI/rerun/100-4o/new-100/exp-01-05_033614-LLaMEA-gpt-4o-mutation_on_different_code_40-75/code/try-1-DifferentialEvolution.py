import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = [func.bounds.lb, func.bounds.ub]
        pop = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = self.pop_size

        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break

                # Adaptive Mutation
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)
                F_adaptive = np.random.uniform(0.4, 0.9)
                mutant = np.clip(pop[a] + F_adaptive * (pop[b] - pop[c]), bounds[0], bounds[1])

                # Adaptive Crossover
                CR_adaptive = np.random.uniform(0.8, 1.0)
                trial = np.where(np.random.rand(self.dim) < CR_adaptive, mutant, pop[i])

                # Selection
                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

        return self.f_opt, self.x_opt