class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget, self.dim))
        for i in range(self.budget):
            for j in range(self.budget):
                idxs = list(range(self.budget))
                idxs.remove(i)
                a, b, c = np.random.choice(idxs, 3, replace=False)
                F_i = np.clip(np.random.normal(self.F, 0.1), 0.1, 0.9)
                CR_i = np.clip(np.random.normal(self.CR, 0.1), 0.1, 0.9)
                mutant = population[a] + F_i * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < CR_i
                trial = np.where(crossover, mutant, population[i])
                f_trial = func(trial)
                if f_trial < func(population[i]):
                    population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

        return self.f_opt, self.x_opt