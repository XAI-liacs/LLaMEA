class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.F_history = np.full(self.pop_size, F)
        
    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))

        for _ in range(self.budget):
            for i in range(self.pop_size):
                idxs = np.random.choice(self.pop_size, size=3, replace=False)
                a, b, c = pop[idxs]

                R = np.random.randint(self.dim)
                trial = np.copy(pop[i])
                for j in range(self.dim):
                    if j == R or np.random.rand() < self.CR:
                        trial[j] = a[j] + self.F_history[i] * (b[j] - c[j])

                trial_fit = func(trial)
                if trial_fit < func(pop[i]):
                    pop[i] = trial

                    if trial_fit < self.f_opt:
                        self.f_opt = trial_fit
                        self.x_opt = trial
                
                self.F_history[i] = self.F + 0.01 * (trial_fit - self.f_opt)

        return self.f_opt, self.x_opt