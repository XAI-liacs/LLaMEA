class DifferentialEvolutionImproved:
    def __init__(self, budget=10000, dim=10, f_lower=0.1, f_upper=0.9, cr_lower=0.1, cr_upper=0.9):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.F = np.random.uniform(f_lower, f_upper)   # Dynamic F parameter
        self.CR = np.random.uniform(cr_lower, cr_upper)  # Dynamic CR parameter
        
    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                F_val = np.random.uniform(0, 1)
                self.F = self.F if F_val < 0.9 else np.random.uniform(0.1, 0.9)  # Adapt F
                
                CR_val = np.random.uniform(0, 1)
                self.CR = self.CR if CR_val < 0.9 else np.random.uniform(0.1, 0.9)  # Adapt CR
                
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