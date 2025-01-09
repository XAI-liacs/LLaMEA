class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for i in range(self.budget):
            for j in range(len(population)):
                idxs = [idx for idx in range(len(population)) if idx != j]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                F_adaptive = max(0.1, min(0.9, np.random.normal(self.F, 0.1)))  # Adaptive F within [0.1, 0.9]
                CR_adaptive = max(0.1, min(1.0, np.random.normal(self.CR, 0.1)))  # Adaptive CR within [0.1, 1.0]
                
                mutant = population[a] + F_adaptive * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < CR_adaptive
                trial = np.where(crossover, mutant, population[j])
                
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    population[j] = trial
                    fitness[j] = f_trial
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
            
        return self.f_opt, self.x_opt