import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=100, F=0.5, Cr=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F
        self.Cr = Cr
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                
                crossover = np.random.rand(self.dim) < self.Cr
                if not np.any(crossover):  # Ensure at least one dimension is crossed
                    crossover[np.random.randint(0, self.dim)] = True
                
                trial = np.where(crossover, mutant, population[i])
                f_trial = func(trial)
                evaluations += 1
                
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                    
                if evaluations >= self.budget:  # Ensure we do not exceed the budget
                    break

            # Adaptation of strategy parameters
            self.F = 0.4 + 0.1 * np.random.rand()
            self.Cr = 0.8 + 0.2 * np.random.rand()

        return self.f_opt, self.x_opt