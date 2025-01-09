class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=50, F_min=0.2, F_max=0.8, F_decay=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.F_min = F_min
        self.F_max = F_max
        self.F_decay = F_decay
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))

        for _ in range(self.budget):
            self.F = max(self.F * self.F_decay, self.F_min)

            for i in range(self.pop_size):
                idxs = np.random.choice(self.pop_size, size=3, replace=False)
                a, b, c = pop[idxs]

                R = np.random.randint(self.dim)
                trial = np.copy(pop[i])
                for j in range(self.dim):
                    if j == R or np.random.rand() < self.CR:
                        trial[j] = a[j] + self.F * (b[j] - c[j])

                trial_fit = func(trial)
                if trial_fit < func(pop[i]):
                    pop[i] = trial

                    if trial_fit < self.f_opt:
                        self.f_opt = trial_fit
                        self.x_opt = trial

        return self.f_opt, self.x_opt