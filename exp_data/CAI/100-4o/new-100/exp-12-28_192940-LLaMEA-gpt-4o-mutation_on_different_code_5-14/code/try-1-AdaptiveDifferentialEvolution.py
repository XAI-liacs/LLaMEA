import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.Inf)
        
    def __call__(self, func):
        evaluations = 0
        func_bounds = (func.bounds.lb, func.bounds.ub)
        
        # Evaluate initial population
        for i in range(self.pop_size):
            self.fitness[i] = func(self.population[i])
            evaluations += 1
            if self.fitness[i] < self.f_opt:
                self.f_opt = self.fitness[i]
                self.x_opt = self.population[i].copy()
        
        # Main loop
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = self.population[indices]
                mutant = np.clip(x1 + self.F * (x2 - x3), func_bounds[0], func_bounds[1])
                
                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, self.population[i])
                
                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()
                
                # Adaptive parameter control
                self.F = 0.5 + np.random.rand() * 0.5
                self.CR = 0.8 + np.random.rand() * 0.2
            
            # Adjust population size
            if evaluations % (self.budget // 10) == 0:
                self.pop_size = max(20, int(self.pop_size * 0.9))
                self.population = self.population[:self.pop_size]
                self.fitness = self.fitness[:self.pop_size]

        return self.f_opt, self.x_opt