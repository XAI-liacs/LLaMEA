class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 10 * self.dim
        bounds = (func.bounds.lb, func.bounds.ub)
        pop = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))
        F_factor = 0.1  # Adaptive control factor for F

        for i in range(self.budget):
            for j in range(pop_size):
                idxs = np.random.choice(pop_size, 3, replace=False)
                a, b, c = pop[idxs]

                F_current = np.clip(np.random.normal(self.F, F_factor), 0, 2)  # Adaptive F value
                mutant = a + F_current * (b - c)
                mutant = np.clip(mutant, bounds[0], bounds[1])

                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, pop[j])

                f_trial = func(trial)
                if f_trial < func(pop[j]):
                    pop[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

        return self.f_opt, self.x_opt