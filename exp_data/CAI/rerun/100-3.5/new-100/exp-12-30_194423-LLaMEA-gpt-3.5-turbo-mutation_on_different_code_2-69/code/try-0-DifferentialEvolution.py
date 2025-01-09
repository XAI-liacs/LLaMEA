import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.population_size = 10
        self.f_opt = np.Inf
        self.x_opt = None

    def mutation(self, population, target_idx):
        candidates = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        return population[a] + self.F * (population[b] - population[c])

    def crossover(self, target, trial):
        crossover_points = np.random.rand(self.dim) < self.CR
        return np.where(crossover_points, trial, target)

    def adapt_mutation(self, iter):
        self.F = 0.5 + 0.3 * np.tanh(iter / 10)
        self.CR = 0.9 - 0.5 * np.tanh(iter / 10)

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.population_size, self.dim))
        
        for i in range(self.budget):
            new_population = []
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
                        
            self.adapt_mutation(i)
            
        return self.f_opt, self.x_opt