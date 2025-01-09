import numpy as np

class DEAPS:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = max(4, 10 * dim)
        self.F = 0.8
        self.CR = 0.9

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        budget_used = self.population_size

        while budget_used < self.budget:
            new_pop = np.empty_like(pop)
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = pop[indices]
                v_i = np.clip(x1 + self.F * (x2 - x3), func.bounds.lb, func.bounds.ub)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, v_i, pop[i])
                f_trial = func(trial)
                budget_used += 1

                if f_trial < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = f_trial
                else:
                    new_pop[i] = pop[i]

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if budget_used >= self.budget:
                    break
            
            pop = new_pop

            # Adaptive population size reduction
            if budget_used % (self.budget // 10) == 0:
                self.population_size = max(4, self.population_size // 2)
                pop = pop[:self.population_size]
                fitness = fitness[:self.population_size]
            
            # Update F and CR adaptively
            self.F = 0.5 + (0.9 - 0.5) * (1 - (self.f_opt / np.min(fitness)))  # Line changed
            self.CR = 0.8 + 0.1 * (1 - (self.f_opt / np.min(fitness)))  # Line changed

        return self.f_opt, self.x_opt