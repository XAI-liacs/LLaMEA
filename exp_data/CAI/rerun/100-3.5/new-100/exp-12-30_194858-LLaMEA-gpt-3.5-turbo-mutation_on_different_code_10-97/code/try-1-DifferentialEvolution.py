class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.8, CR=0.9, F_lower=0.2, F_upper=0.9, CR_lower=0.1, CR_upper=1.0):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.F_lower = F_lower
        self.F_upper = F_upper
        self.CR_lower = CR_lower
        self.CR_upper = CR_upper
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        F_history = np.zeros(self.budget)
        CR_history = np.zeros(self.budget)
        
        for i in range(self.budget):
            for j in range(len(population)):
                indices = [idx for idx in range(len(population)) if idx != j]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # Adaptive control of F and CR parameters
                F = np.random.uniform(self.F_lower, self.F_upper)
                CR = np.random.uniform(self.CR_lower, self.CR_upper)
                
                mutant = population[a] + F * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[j])
                
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    population[j] = trial
                    fitness[j] = f_trial
                
                if fitness[j] < self.f_opt:
                    self.f_opt = fitness[j]
                    self.x_opt = population[j]
                    
                F_history[i] = F
                CR_history[i] = CR
                
        return self.f_opt, self.x_opt