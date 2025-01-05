import numpy as np

class DEACM:
    def __init__(self, budget=10000, dim=10, init_pop_size=50, F=0.5, CR=0.5):
        self.budget = budget
        self.dim = dim
        self.init_pop_size = init_pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = self.init_pop_size
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = pop[best_idx].copy()

        evals = pop_size
        chaotic_sequence = np.random.rand(self.budget)

        while evals < self.budget:
            for i in range(pop_size):
                indices = list(range(pop_size))
                indices.remove(i)
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()

                if evals >= self.budget:
                    break
            
            # Reduce population size adaptively
            if evals < self.budget / 2:
                pop_size = self.init_pop_size - int((self.init_pop_size / 2) * (evals / (self.budget / 2)))
            
            # Chaotic maps for F and CR
            self.F = 0.5 + 0.3 * np.sin(chaotic_sequence[evals] * np.pi)
            self.CR = 0.5 + 0.3 * np.cos(chaotic_sequence[evals] * np.pi)
            
        return self.f_opt, self.x_opt