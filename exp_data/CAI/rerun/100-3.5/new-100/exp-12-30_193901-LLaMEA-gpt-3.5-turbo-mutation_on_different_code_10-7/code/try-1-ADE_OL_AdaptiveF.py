class ADE_OL_AdaptiveF:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_lb=0.1, F_ub=0.9, adapt_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.F_lb = F_lb
        self.F_ub = F_ub
        self.adapt_rate = adapt_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = func.bounds
        pop_size = 10 * self.dim
        population = np.random.uniform(bounds.lb, bounds.ub, size=(pop_size, self.dim))

        for i in range(self.budget):
            for j in range(pop_size):
                target = population[j]
                
                # Mutation with dynamic adaptation of F
                F_dynamic = np.clip(self.F * np.exp(-self.adapt_rate*i), self.F_lb, self.F_ub)
                idxs = np.random.choice(pop_size, size=3, replace=False)
                a, b, c = population[idxs]
                mutant = np.clip(target + F_dynamic * (a - target) + F_dynamic * (b - c), bounds.lb, bounds.ub)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, target)
                
                # Selection
                f_target = func(target)
                f_trial = func(trial)
                if f_trial < f_target:
                    population[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    population[j] = target
            
        return self.f_opt, self.x_opt