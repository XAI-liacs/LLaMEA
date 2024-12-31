class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None
        self.F_history = [F]
        self.CR_history = [CR]

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        best_idx = np.argmin([func(x) for x in population])
        self.f_opt = func(population[best_idx])
        self.x_opt = population[best_idx].copy()

        for i in range(self.budget):
            for j in range(len(population)):
                idxs = [idx for idx in range(len(population)) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F_current = np.random.normal(np.mean(self.F_history), 0.1)
                CR_current = np.random.normal(np.mean(self.CR_history), 0.1)
                mutant = np.clip(a + F_current * (b - c), func.bounds.lb, func.bounds.ub)
                cross_points = np.random.rand(self.dim) < CR_current
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = population[j].copy()
                trial[cross_points] = mutant[cross_points]
                f_trial = func(trial)
                if f_trial < func(population[j]):
                    population[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()
                        self.F_history.append(F_current)
                        self.CR_history.append(CR_current)

        return self.f_opt, self.x_opt