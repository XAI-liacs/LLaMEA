import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, population_size=50):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            new_population = np.zeros_like(population)
            
            for j in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != j]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = population[a] + self.F * (population[b] - population[c])
                crossover_prob = np.random.uniform(0, 1, self.dim) < self.CR
                trial = np.where(crossover_prob, mutant, population[j])
                
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    fitness[j] = f_trial
                    population[j] = trial
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
            
        return self.f_opt, self.x_opt