class ImprovedDifferentialEvolution(DifferentialEvolution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.F_history = [self.F]

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        for _ in range(self.budget):
            for i in range(self.pop_size):
                a, b, c = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                F_current = np.clip(np.random.normal(self.F, 0.1), 0, 2)  # Adaptive F parameter
                mutant = pop[a] + F_current * (pop[b] - pop[c])
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, pop[i])
                f = func(trial)
                if f < func(pop[i]):
                    pop[i] = trial
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
                self.F_history.append(F_current)
        return self.f_opt, self.x_opt