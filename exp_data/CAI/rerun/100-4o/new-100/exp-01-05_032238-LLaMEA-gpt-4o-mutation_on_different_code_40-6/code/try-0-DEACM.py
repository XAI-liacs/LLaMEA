import numpy as np

class DEACM:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = pop[best_idx].copy()

        evals = self.pop_size  # Function evaluations used

        while evals < self.budget:
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
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
                        
            # Adaptive strategies for F and CR
            self.F = 0.5 + 0.3 * np.sin(0.5 * np.pi * evals / self.budget)  # Vary between 0.2 and 0.8
            self.CR = 0.5 + 0.3 * np.cos(0.5 * np.pi * evals / self.budget)  # Vary between 0.2 and 0.8
            
        return self.f_opt, self.x_opt