import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= self.pop_size

        for i in range(self.budget):
            for j in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.bounds[0], self.bounds[1])
                
                crossover = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                
                trial = np.where(crossover, mutant, population[j])
                f_trial = func(trial)
                
                if f_trial < fitness[j]:
                    population[j] = trial
                    fitness[j] = f_trial
                    
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
            
            # Adaptation strategy for crossover and mutation rates
            self.mutation_factor = np.clip(self.mutation_factor + np.random.normal(0, 0.1), 0.1, 1.0)
            self.crossover_rate = np.clip(self.crossover_rate + np.random.normal(0, 0.1), 0.1, 1.0)

        return self.f_opt, self.x_opt