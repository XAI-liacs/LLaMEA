class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.population_size = 10
        self.f_opt = np.Inf
        self.x_opt = None

    def adapt_population_size(self, fitnesses):
        diversity = np.std(fitnesses)
        if diversity < 1e-6:  # Ensuring diversity to prevent stagnation
            self.population_size = min(20, self.population_size * 2)  # Double the population size
        elif diversity > 1:
            self.population_size = max(5, self.population_size // 2)  # Halve the population size

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.population_size, self.dim))
        
        for i in range(self.budget):
            new_population = []
            fitnesses = []
            for j in range(self.population_size):
                x_target = population[j]
                x_trial = self.crossover(x_target, self.mutation(population, j))
                f_target = func(x_target)
                f_trial = func(x_trial)
                
                if f_trial < f_target:
                    population[j] = x_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = x_trial
                        
                fitnesses.append(f_trial)
            
            self.adapt_mutation(i)
            self.adapt_population_size(fitnesses)
            
        return self.f_opt, self.x_opt