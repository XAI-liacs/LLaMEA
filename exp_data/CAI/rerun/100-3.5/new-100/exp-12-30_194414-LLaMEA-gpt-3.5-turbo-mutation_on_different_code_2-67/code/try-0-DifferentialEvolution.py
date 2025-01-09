import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def mutation(self, population, target_idx):
        candidates = list(range(self.pop_size))
        candidates.remove(target_idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        return population[a] + self.F * (population[b] - population[c])

    def crossover(self, target, mutant):
        trial = np.copy(target)
        j_rand = np.random.randint(0, self.dim)
        for j in range(self.dim):
            if np.random.rand() > self.CR and j != j_rand:
                trial[j] = mutant[j]
        return trial

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        
        for i in range(self.budget // self.pop_size):
            for j in range(self.pop_size):
                x = population[j]
                v = self.mutation(population, j)
                u = self.crossover(x, v)
                f_x, f_u = func(x), func(u)
                
                if f_u < f_x:
                    population[j] = u
                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u
        
        return self.f_opt, self.x_opt