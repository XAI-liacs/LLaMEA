class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.F_init = 0.5  # Initial value for Differential weight
        self.CR_init = 0.9  # Initial value for Crossover rate
        self.F_lb = 0.1  # Lower bound for Differential weight
        self.F_ub = 0.9  # Upper bound for Differential weight
        self.CR_lb = 0.1  # Lower bound for Crossover rate
        self.CR_ub = 0.9  # Upper bound for Crossover rate

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        F = self.F_init
        CR = self.CR_init
        
        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                F = np.clip(np.random.normal(F, 0.1), self.F_lb, self.F_ub)
                CR = np.clip(np.random.normal(CR, 0.1), self.CR_lb, self.CR_ub)
                
                mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)
                
                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, population[j])
                
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    population[j] = trial
                    fitness[j] = f_trial
                    
                if fitness[j] < self.f_opt:
                    self.f_opt = fitness[j]
                    self.x_opt = population[j]

        return self.f_opt, self.x_opt