import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        pop = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        budget_spent = self.pop_size
        
        while budget_spent < self.budget:
            for i in range(self.pop_size):
                candidates = list(range(self.pop_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = np.clip(pop[a] + self.F * (pop[b] - pop[c]), bounds[0], bounds[1])
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                f_trial = func(trial)
                budget_spent += 1
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                
                if budget_spent >= self.budget:
                    break
            
            # Adapt F and CR
            self.F = np.random.normal(0.5, 0.3)
            self.CR = np.random.normal(0.5, 0.1)
            self.F = np.clip(self.F, 0.4, 1.0)
            self.CR = np.clip(self.CR, 0.1, 1.0)
        
        return self.f_opt, self.x_opt