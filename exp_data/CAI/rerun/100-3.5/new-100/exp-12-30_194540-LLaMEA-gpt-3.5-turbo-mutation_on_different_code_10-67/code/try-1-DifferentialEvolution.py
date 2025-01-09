class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            candidate = population[i]
            idxs = [idx for idx in range(self.budget) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            F_i = np.clip(self.F + 0.5 * np.tanh((self.budget - i) / self.budget), 0.1, 0.9) # Dynamic F
            CR_i = np.clip(self.CR - 0.5 * np.tanh((self.budget - i) / self.budget), 0.1, 0.9) # Dynamic CR
            mutant = np.clip(a + F_i * (b - c), func.bounds.lb, func.bounds.ub)
            crossover_mask = np.random.rand(self.dim) < CR_i
            trial = np.where(crossover_mask, mutant, candidate)
            f = func(trial)
            if f < func(candidate):
                population[i] = trial
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
        return self.f_opt, self.x_opt