class DifferentialEvolutionImproved:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=50, F_min=0.2, F_max=0.8, CR_min=0.1, CR_max=1.0, diversity_factor=0.5):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max
        self.diversity_factor = diversity_factor
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        def bound_check(x):
            return np.clip(x, func.bounds.lb, func.bounds.ub)

        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        prev_diversity = np.linalg.norm(np.std(pop, axis=0))

        for _ in range(self.budget):
            new_pop = np.zeros_like(pop)
            for i in range(self.pop_size):
                a, b, c = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                F = np.clip(np.random.normal(self.F, self.diversity_factor), self.F_min, self.F_max)
                CR = np.clip(np.random.normal(self.CR, self.diversity_factor), self.CR_min, self.CR_max)
                mutant = bound_check(pop[a] + F * (pop[b] - pop[c]))
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, pop[i])
                f_trial = func(trial)
                if f_trial < func(pop[i]):
                    pop[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
            diversity = np.linalg.norm(np.std(pop, axis=0))
            self.diversity_factor = 0.5 * self.diversity_factor + 0.5 * (diversity - prev_diversity)
            prev_diversity = diversity

        return self.f_opt, self.x_opt