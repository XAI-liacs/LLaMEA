import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.7, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None
    
    def mutate(self, population, target_idx):
        idxs = list(range(len(population)))
        idxs.remove(target_idx)
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant_vector = population[a] + self.F * (population[b] - population[c])
        return mutant_vector
    
    def crossover(self, target_vector, mutant_vector):
        trial_vector = np.copy(target_vector)
        for i in range(len(target_vector)):
            if np.random.rand() > self.CR:
                trial_vector[i] = mutant_vector[i]
        return trial_vector
    
    def tournament_selection(self, trial_vector, target_vector, func):
        target_fitness = func(target_vector)
        trial_fitness = func(trial_vector)
        if trial_fitness < target_fitness:
            return trial_vector, trial_fitness
        else:
            return target_vector, target_fitness
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        
        for i in range(self.budget):
            for j in range(len(population)):
                mutant_vector = self.mutate(population, j)
                trial_vector = self.crossover(population[j], mutant_vector)
                population[j], fitness = self.tournament_selection(trial_vector, population[j], func)
                
                if fitness < self.f_opt:
                    self.f_opt = fitness
                    self.x_opt = population[j]
        
        return self.f_opt, self.x_opt