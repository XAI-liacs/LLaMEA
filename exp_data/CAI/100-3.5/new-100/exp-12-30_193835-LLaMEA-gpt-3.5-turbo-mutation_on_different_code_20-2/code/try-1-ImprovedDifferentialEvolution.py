import numpy as np

class ImprovedDifferentialEvolution(DifferentialEvolution):
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=50, F_min=0.2, F_max=0.8, F_decay=0.99):
        super().__init__(budget, dim, F, CR, pop_size)
        self.F_min = F_min
        self.F_max = F_max
        self.F_decay = F_decay

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        F_current = self.F

        for _ in range(self.budget // self.pop_size):
            for idx, target in enumerate(population):
                mutant = self.mutate(population, idx, F_current)
                trial = self.crossover(target, mutant)
                
                f_target = func(target)
                f_trial = func(trial)
                
                if f_trial < f_target:
                    population[idx] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

            F_current = max(self.F_min, min(self.F_max, F_current * self.F_decay))  # Adaptive control of F

        return self.f_opt, self.x_opt