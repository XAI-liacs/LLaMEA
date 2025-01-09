import numpy as np

class DEPSO:
    def __init__(self, budget, dim, pop_size=30, w=0.5, c1=1.5, c2=1.5, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.f = f
        self.cr = cr

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, pop)
        best_idx = np.argmin(fitness)
        best_pos = pop[best_idx].copy()
        
        for _ in range(self.budget - self.pop_size):
            new_pop = np.zeros((self.pop_size, self.dim))
            for i in range(self.pop_size):
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = pop[idxs]
                mutant = pop[i] + self.f * (a - pop[i]) + self.f * (b - c)
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, pop[i])
                pbest = pop[best_idx]
                vel = self.w * (pop[i] + self.c1 * np.random.rand() * (pbest - pop[i]) + self.c2 * np.random.rand() * (best_pos - pop[i]) - pop[i])
                new_pop[i] = trial + vel
            new_fitness = np.apply_along_axis(func, 1, new_pop)
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    pop[i] = new_pop[i]
                    fitness[i] = new_fitness[i]
                    if new_fitness[i] < func(best_pos):
                        best_pos = new_pop[i].copy()
        return best_pos