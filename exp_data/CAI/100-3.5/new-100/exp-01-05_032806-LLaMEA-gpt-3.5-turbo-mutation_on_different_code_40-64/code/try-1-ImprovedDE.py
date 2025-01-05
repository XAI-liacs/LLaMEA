class ImprovedDE(DifferentialEvolution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.F_lb = 0.3
        self.F_ub = 0.8
        self.CR_lb = 0.1
        self.CR_ub = 0.9

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        
        for i in range(self.budget):
            F = np.random.uniform(self.F_lb, self.F_ub)
            CR = np.random.uniform(self.CR_lb, self.CR_ub)
            
            for j in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = population[a] + F * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[j])
                
                f_trial = func(trial)
                if f_trial < func(population[j]):
                    population[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
            
        return self.f_opt, self.x_opt