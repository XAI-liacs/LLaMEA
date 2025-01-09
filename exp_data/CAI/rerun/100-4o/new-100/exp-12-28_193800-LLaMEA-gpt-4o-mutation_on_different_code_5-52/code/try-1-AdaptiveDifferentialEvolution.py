import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=20, F_init=0.5, CR_init=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F_init
        self.CR = CR_init
        self.f_opt = np.Inf
        self.x_opt = None
        self.success_F = []
    
    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                idxs = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                a, b, c = population[idxs]
                F_local = np.mean(self.success_F[-5:]) if len(self.success_F) >= 5 else self.F  # Adjusted scale factor
                mutant = np.clip(a + F_local * (b - c), bounds[0], bounds[1])
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                f_trial = func(trial)
                eval_count += 1
                
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    self.success_F.append(F_local)  # Store successful F values
                    
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                
                if eval_count >= self.budget:
                    break

            # Adaptive control of F and CR
            self.F = np.clip(self.F + 0.1 * (np.random.rand() - 0.5), 0.1, 0.9)
            self.CR = np.clip(self.CR + 0.1 * (np.random.rand() - 0.5), 0.1, 0.9)

        return self.f_opt, self.x_opt