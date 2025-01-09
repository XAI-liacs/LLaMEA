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
            for j in range(pop_size):
                a, b, c = np.random.choice(population, 3, replace=False)
                mutant = a + self.F * (b - c)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[j])
                
                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
                    population[j] = trial
            
            if i % 100 == 0:  # Adaptive adjustment every 100 iterations
                successful_rate = np.sum(np.array([func(population[k]) < func(population[k-1]) for k in range(1, pop_size)])) / (pop_size - 1)
                self.F = max(0.1, min(1.0, self.F + 0.1 * (successful_rate - 0.5)))
                self.CR = max(0.1, min(1.0, self.CR + 0.1 * (successful_rate - 0.5)))
                
        return self.f_opt, self.x_opt