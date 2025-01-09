import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=100, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Initialize population
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.full(self.population_size, np.Inf)
        
        # Evaluate initial population
        for i in range(self.population_size):
            fitness[i] = func(population[i])
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = population[i]

        evals = self.population_size

        while evals < self.budget:
            # Adaptive parameter control
            F_adapt = self.F * (1 - evals / self.budget)
            CR_adapt = self.CR * (1 - evals / self.budget)
            
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices[0]], population[indices[1]], population[indices[2]]
                mutant = np.clip(a + F_adapt * (b - c), lb, ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < CR_adapt
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial
                
                if evals >= self.budget:
                    break

        return self.f_opt, self.x_opt