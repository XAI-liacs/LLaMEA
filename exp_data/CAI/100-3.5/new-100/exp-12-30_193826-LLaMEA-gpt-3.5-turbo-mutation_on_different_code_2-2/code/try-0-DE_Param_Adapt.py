import numpy as np

class DE_Param_Adapt:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 10 * self.dim
        bounds = (func.bounds.lb, func.bounds.ub)
        pop = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        
        for _ in range(self.budget // pop_size):
            for i in range(pop_size):
                idxs = np.arange(pop_size)
                np.random.shuffle(idxs)
                a, b, c = pop[np.random.choice(idxs[:3], 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])
                
                j_rand = np.random.randint(self.dim)
                trial = np.array([mutant[j] if np.random.uniform() < self.CR or j == j_rand else pop[i, j]
                                  for j in range(self.dim)])
                
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    pop[i], fitness[i] = trial, f_trial
                    if f_trial < self.f_opt:
                        self.f_opt, self.x_opt = f_trial, trial
        
        return self.f_opt, self.x_opt