class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_min=0.2, F_max=0.8):
        self.budget = budget
        self.dim = dim
        self.F_min = F_min
        self.F_max = F_max
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        F_val = self.F
        for i in range(self.budget):
            idxs = list(range(self.budget))
            idxs.remove(i)
            a, b, c = np.random.choice(idxs, 3, replace=False)
            mutant = population[a] + F_val * (population[b] - population[c])
            cross_points = np.random.rand(self.dim) < self.CR
            trial = np.where(cross_points, mutant, population[i])
            f = func(trial)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = trial
                population[i] = trial
            if np.random.rand() < 0.1:  # Adjust F with 10% probability
                F_val = self.F_min + np.random.rand() * (self.F_max - self.F_min)
        return self.f_opt, self.x_opt