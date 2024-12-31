import numpy as np

class DEACM:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 20
        self.CR = 0.9
        self.F = 0.8
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size

        while eval_count < self.budget:
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                # Adaptive mutation factor F
                F_adaptive = (np.random.rand() * 0.5 + 0.5) * self.F
                
                mutant = np.clip(a + F_adaptive * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                f_trial = func(trial)
                eval_count += 1
                
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                
                if eval_count >= self.budget:
                    break

        return self.f_opt, self.x_opt