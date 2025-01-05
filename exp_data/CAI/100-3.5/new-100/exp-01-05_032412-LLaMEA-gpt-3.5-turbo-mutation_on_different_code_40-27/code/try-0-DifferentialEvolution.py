import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population_size = 10
        F = 0.5
        CR = 0.9
        mutation_strategies = ['best', 'rand', 'current-to-best']
        
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(population_size, self.dim))
        for i in range(self.budget):
            for j in range(population_size):
                target = population[j]
                
                r1, r2, r3 = np.random.choice(population, 3, replace=False)
                if np.random.rand() < CR:
                    mutation_strategy = np.random.choice(mutation_strategies)
                    if mutation_strategy == 'best':
                        donor = population[np.argmin([func(x) for x in population])]
                    elif mutation_strategy == 'rand':
                        donor = r1 + F * (r2 - r3)
                    else:
                        donor = target + F * (r1 - target) + F * (r2 - r3)
                    
                    trial = np.where(np.random.rand(self.dim) < CR, donor, target)
                    
                    f = func(trial)
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial
                    
        return self.f_opt, self.x_opt