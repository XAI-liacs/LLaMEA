import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=30):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def evolve_population(self, population, func):
        new_population = []
        for i in range(self.pop_size):
            target = population[i]
            a, b, c = np.random.choice(population, 3, replace=False)
            F_mutant = np.random.normal(self.F, 0.1) if np.random.rand() > 0.1 else self.F
            CR_crossover = np.random.normal(self.CR, 0.1) if np.random.rand() > 0.1 else self.CR
            mutant = a + F_mutant * (b - c)
            crossover = np.random.rand(self.dim) < CR_crossover
            trial = np.where(crossover, mutant, target)
            f_trial = func(trial)
            if f_trial < func(target):
                new_population.append(trial)
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
            else:
                new_population.append(target)
        return new_population

    def __call__(self, func):
        population = [np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim) for _ in range(self.pop_size)]
        
        for i in range(self.budget // self.pop_size):
            population = self.evolve_population(population, func)
            
        return self.f_opt, self.x_opt