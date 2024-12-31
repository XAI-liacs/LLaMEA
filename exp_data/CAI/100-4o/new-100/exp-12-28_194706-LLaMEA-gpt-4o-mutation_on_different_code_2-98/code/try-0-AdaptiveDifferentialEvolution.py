import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.population = np.random.uniform(-5, 5, (self.population_size, dim))
        self.F_base = 0.5
        self.CR_base = 0.9

    def __call__(self, func):
        func_evals = 0
        
        while func_evals < self.budget:
            F = self.F_base + 0.1 * np.random.normal()
            CR = self.CR_base + 0.1 * np.random.normal()
            F = np.clip(F, 0.1, 0.9)
            CR = np.clip(CR, 0.1, 0.9)

            new_population = np.copy(self.population)
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[indices]
                
                mutant = np.clip(a + F * (b - c), -5, 5)
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, self.population[i])
                
                f_trial = func(trial)
                func_evals += 1
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                if f_trial < func(self.population[i]):
                    new_population[i] = trial

                if func_evals >= self.budget:
                    break

            self.population = new_population

        return self.f_opt, self.x_opt