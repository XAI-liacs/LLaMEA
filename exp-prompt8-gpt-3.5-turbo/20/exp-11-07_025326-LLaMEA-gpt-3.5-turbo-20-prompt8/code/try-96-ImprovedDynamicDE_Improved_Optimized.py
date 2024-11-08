import numpy as np

class ImprovedDynamicDE_Improved_Optimized:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.CR = 0.5  # Crossover rate
        self.F_min = 0.2  # Minimum scaling factor
        self.F_max = 0.8  # Maximum scaling factor

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (int(0.8*self.budget), self.dim))  # Reduced population initialization size
        best_fitness = func(population[0])
        
        for _ in range(self.budget):
            F = self.F_min + np.random.rand() * (self.F_max - self.F_min)
            for i in range(len(population)):  # Use length of population instead of budget
                idx = np.delete(np.arange(len(population)), i)
                a, b, c = population[np.random.choice(idx, 3, replace=False)]
                j_rand = np.random.randint(self.dim)
                mutant = np.clip(a + F * (b - c), -5.0, 5.0)
                
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                f_trial = func(trial)
                
                if f_trial < best_fitness:
                    population[i] = trial
                    best_fitness = f_trial
        
        return population[np.argmin([func(individual) for individual in population])]