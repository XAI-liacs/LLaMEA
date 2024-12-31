import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None
    
    def select_parents(self, pop, idx):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        return pop[np.random.choice(indices, 3, replace=False)]

    def mutation(self, x, a, b, c):
        return np.clip(a + self.F * (b - c), -5.0, 5.0)
    
    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        return np.where(cross_points, mutant, target)
    
    def local_search(self, x, func):
        step_size = 0.01
        for _ in range(10):
            step = np.random.uniform(-step_size, step_size, self.dim)
            neighbor = np.clip(x + step, -5.0, 5.0)
            if func(neighbor) < func(x):
                x = neighbor
        return x
    
    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        for _ in range(self.budget - self.pop_size):
            for i in range(self.pop_size):
                a, b, c = self.select_parents(pop, i)
                mutant = self.mutation(pop[i], a, b, c)
                trial = self.crossover(pop[i], mutant)
                trial = self.local_search(trial, func)

                f_trial = func(trial)
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    self.F = min(1, self.F + 0.01)  # Adjust F adaptively

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

        return self.f_opt, self.x_opt