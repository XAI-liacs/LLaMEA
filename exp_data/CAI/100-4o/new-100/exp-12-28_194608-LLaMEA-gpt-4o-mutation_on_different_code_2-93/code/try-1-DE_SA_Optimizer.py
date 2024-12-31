import numpy as np

class DE_SA_Optimizer:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20  # Population size for Differential Evolution
        self.f_opt = np.Inf
        self.x_opt = None
    
    def differential_evolution(self, pop, func, bounds):
        F = 0.8  # Differential weight
        CR_base = 0.9  # Base crossover probability
        new_pop = np.zeros_like(pop)
        for i in range(self.pop_size):
            CR = CR_base * (1 - (self.budget // self.pop_size) / self.budget)  # Dynamic CR
            indices = np.arange(self.pop_size)
            indices = indices[indices != i]
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, pop[i])
            f_trial = func(trial)
            f_target = func(pop[i])
            if f_trial < f_target:
                new_pop[i] = trial
                if f_trial < self.f_opt:
                    self.f_opt, self.x_opt = f_trial, trial
            else:
                new_pop[i] = pop[i]
        return new_pop

    def simulated_annealing(self, pop, func, bounds, temp):
        for i in range(self.pop_size):
            neighbor = pop[i] + np.random.uniform(-0.1, 0.1, self.dim) * temp
            neighbor = np.clip(neighbor, bounds.lb, bounds.ub)
            f_neighbor = func(neighbor)
            f_current = func(pop[i])
            if f_neighbor < f_current or np.random.rand() < np.exp((f_current - f_neighbor) / temp):
                pop[i] = neighbor
                if f_neighbor < self.f_opt:
                    self.f_opt, self.x_opt = f_neighbor, neighbor
        return pop
    
    def __call__(self, func):
        bounds = func.bounds
        pop = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
        for _ in range(self.budget // self.pop_size):
            temp = 1.0 - (_ / (self.budget // self.pop_size))  # Cooling schedule for SA
            pop = self.differential_evolution(pop, func, bounds)
            pop = self.simulated_annealing(pop, func, bounds, temp)
        return self.f_opt, self.x_opt