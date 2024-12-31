class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 10 * self.dim
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))

        for i in range(self.budget):
            F_current = self.F * (0.5 + 0.5 * np.random.rand())  # Adaptive control of F parameter
            CR_current = self.CR * (0.1 + 0.9 * np.random.rand())  # Adaptive control of CR parameter

            for j in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F_current * (b - c), bounds[0], bounds[1])

                cross_points = np.random.rand(self.dim) < CR_current
                trial = np.where(cross_points, mutant, population[j])

                f_trial = func(trial)
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                    population[j] = trial

        return self.f_opt, self.x_opt