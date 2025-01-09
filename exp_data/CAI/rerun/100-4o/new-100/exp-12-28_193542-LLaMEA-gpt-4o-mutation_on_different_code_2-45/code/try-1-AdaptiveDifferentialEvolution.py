import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20 + int(3 * np.log(self.dim))  # A common heuristic for DE
        self.F = 0.5  # mutation factor
        self.CR = 0.9  # crossover probability
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def __call__(self, func):
        func_evals = 0
        f_values = np.apply_along_axis(func, 1, self.population)
        func_evals += self.population_size
        
        for i in range(self.population_size):
            if f_values[i] < self.f_opt:
                self.f_opt = f_values[i]
                self.x_opt = self.population[i]
        
        while func_evals < self.budget:
            sorted_indices = np.argsort(f_values)
            elite_count = int(0.1 * self.population_size)  # Keep top 10% unchanged
            for i in range(self.population_size):
                if func_evals >= self.budget or i < elite_count:
                    continue
                
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                
                mutant = np.clip(a + self.F * (b - c), -5.0, 5.0)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])
                
                f_trial = func(trial)
                func_evals += 1

                if f_trial < f_values[i]:
                    self.population[i] = trial
                    f_values[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                # Dynamic adjustment of F and CR
                self.F = 0.5 + 0.3 * np.random.rand()
                self.CR = 0.8 + 0.1 * np.random.rand()

        return self.f_opt, self.x_opt