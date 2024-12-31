import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.f_opt = np.Inf
        self.x_opt = None

    def mutate(self, target, population):
        candidates = np.copy(population)
        candidates = np.delete(candidates, target, axis=0)
        np.random.shuffle(candidates)
        a, b, c = candidates[:3]
        mutant_vector = population[a] + self.F * (population[b] - population[c])
        return mutant_vector

    def crossover(self, target, mutant_vector):
        trial_vector = np.copy(self.population[target])
        for i in range(self.dim):
            if np.random.uniform() > self.CR:
                trial_vector[i] = mutant_vector[i]
        return trial_vector

    def __call__(self, func):
        for i in range(self.budget):
            for target in range(len(self.population)):
                x_mutant = self.mutate(target, self.population)
                x_trial = self.crossover(target, x_mutant)
                f_trial = func(x_trial)
                if f_trial < func(self.population[target]):
                    self.population[target] = x_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = x_trial
        
        return self.f_opt, self.x_opt