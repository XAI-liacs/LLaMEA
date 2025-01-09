class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_lb=0.1, F_ub=0.9, CR_lb=0.1, CR_ub=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.F_lb = F_lb
        self.F_ub = F_ub
        self.CR_lb = CR_lb
        self.CR_ub = CR_ub
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim, self.dim))
        for _ in range(self.budget):
            F = np.random.uniform(self.F_lb, self.F_ub)
            CR = np.random.uniform(self.CR_lb, self.CR_ub)
            for i in range(self.dim):
                idxs = [idx for idx in range(self.dim) if idx != i]
                a, b, c = np.random.choice(population[idxs], 3, replace=False)
                mutant = a + F * (b - c)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, population[i])
                f = func(trial)
                if f < func(population[i]):
                    population[i] = trial
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial

        return self.f_opt, self.x_opt