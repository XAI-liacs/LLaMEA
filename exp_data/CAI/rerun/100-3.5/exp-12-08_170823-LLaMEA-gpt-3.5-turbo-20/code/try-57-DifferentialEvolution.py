class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.CR = 0.9  # Crossover rate
        self.F = 0.5   # Differential weight
        self.CR_history = np.full(self.budget, self.CR)  # Track CR
        self.F_history = np.full(self.budget, self.F)    # Track F

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])

        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                # Adaptive control parameters
                self.CR = np.clip(np.mean(self.CR_history), 0, 1)
                self.F = np.mean(self.F_history)
                self.CR_history[i] = self.CR
                self.F_history[i] = self.F

                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)

                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[j])

                f_trial = func(trial)
                if f_trial < fitness[j]:
                    population[j] = trial
                    fitness[j] = f_trial

                if fitness[j] < self.f_opt:
                    self.f_opt = fitness[j]
                    self.x_opt = population[j]

        return self.f_opt, self.x_opt