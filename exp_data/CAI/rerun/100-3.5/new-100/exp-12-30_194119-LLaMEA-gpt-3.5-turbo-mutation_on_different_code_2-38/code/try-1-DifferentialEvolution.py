import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 10 * self.dim
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        
        for _ in range(self.budget // pop_size):
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                F_current = self.F * np.random.uniform(0.5, 1.0)  # Dynamic F
                mutant = np.clip(a + F_current * (b - c), -5.0, 5.0)
                
                j_rand = np.random.randint(self.dim)
                trial = [mutant[j] if np.random.rand() < self.CR or j == j_rand else pop[i, j] for j in range(self.dim)]
                
                f_trial = func(trial)
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                    pop[i] = trial
                    
        return self.f_opt, self.x_opt