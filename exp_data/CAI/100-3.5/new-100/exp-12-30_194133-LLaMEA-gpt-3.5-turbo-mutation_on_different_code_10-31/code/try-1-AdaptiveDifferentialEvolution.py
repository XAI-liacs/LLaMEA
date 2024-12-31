class AdaptiveDifferentialEvolution(DifferentialEvolution):
    def __init__(self, budget=10000, dim=10, pop_size=50):
        super().__init__(budget, dim, pop_size=50)
        self.F_lb = 0.4
        self.F_ub = 0.9
        self.CR_lb = 0.7
        self.CR_ub = 1.0

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        F = np.random.uniform(self.F_lb, self.F_ub)
        CR = np.random.uniform(self.CR_lb, self.CR_ub)

        for i in range(self.budget):
            for j in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

                mutant = pop[j] + F * (a - pop[j]) + F * (b - c)
                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, pop[j])

                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
                    pop[j] = trial

            # Adaptively update F and CR based on function landscape
            mean_f = np.mean([func(x) for x in pop])
            if mean_f < self.f_opt:
                F = min(self.F_ub, F * 1.2)  # Increase F
                CR = max(self.CR_lb, CR * 0.9)  # Decrease CR
            else:
                F = max(self.F_lb, F * 0.8)  # Decrease F
                CR = min(self.CR_ub, CR * 1.1)  # Increase CR

        return self.f_opt, self.x_opt