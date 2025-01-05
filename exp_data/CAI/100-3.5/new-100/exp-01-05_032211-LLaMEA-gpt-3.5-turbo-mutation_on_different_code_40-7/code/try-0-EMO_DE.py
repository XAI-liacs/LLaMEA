import numpy as np

class EMO_DE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        def mutate(x_r1, x_r2, x_r3):
            return x_r1 + self.F * (x_r2 - x_r3)

        def crossover(x, v):
            j_rand = np.random.randint(self.dim)
            u = np.array([v[i] if i == j_rand or np.random.rand() < self.CR else x[i] for i in range(self.dim)])
            return u

        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))

        for i in range(self.budget // self.pop_size):
            for j in range(self.pop_size):
                x = population[j]
                idx_r1, idx_r2, idx_r3 = np.random.choice(np.delete(np.arange(self.pop_size), j), 3, replace=False)
                x_r1, x_r2, x_r3 = population[idx_r1], population[idx_r2], population[idx_r3]
                
                v = mutate(x_r1, x_r2, x_r3)
                u = crossover(x, v)
                
                f_u = func(u)
                if f_u < self.f_opt:
                    self.f_opt = f_u
                    self.x_opt = u
                population[j] = u if f_u < func(x) else x

        return self.f_opt, self.x_opt