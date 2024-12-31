import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        
        # Adaptation parameters
        self.F_min = 0.5
        self.F_max = 0.9
        self.CR_min = 0.1
        self.CR_max = 0.9

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        self.budget -= self.pop_size
        
        # Identify the best initial individual
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx].copy()
        
        while self.budget > 0:
            new_population = []
            for i in range(self.pop_size):
                # Select three unique indices different from i
                idxs = list(range(self.pop_size))
                idxs.remove(i)
                a, b, c = np.random.choice(idxs, 3, replace=False)
                
                # Adaptively calculate F and CR
                F = np.random.uniform(self.F_min, self.F_max)
                CR = np.random.uniform(self.CR_min, self.CR_max)
                
                # Mutation
                mutant = population[a] + F * (population[b] - population[c])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                # Crossover
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                self.budget -= 1
                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                    fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial
                else:
                    new_population.append(population[i])
                
                if self.budget <= 0:
                    break
            
            if self.budget <= 0:
                break
            
            population = np.array(new_population)
        
        return self.f_opt, self.x_opt