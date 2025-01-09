import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def mutate(self, population, target_index):
        r1, r2, r3 = np.random.choice(len(population), 3, replace=False)
        mutated_vector = population[r1] + self.F * (population[r2] - population[r3])
        return mutated_vector

    def crossover(self, target_vector, mutated_vector):
        trial_vector = np.copy(target_vector)
        for i in range(len(target_vector)):
            if np.random.rand() > self.CR:
                trial_vector[i] = mutated_vector[i]
        return trial_vector

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim, self.dim))
        for i in range(self.budget):
            for j in range(self.dim):
                target_vector = population[j]
                mutated_vector = self.mutate(population, j)
                trial_vector = self.crossover(target_vector, mutated_vector)
                f = func(trial_vector)
                if f < func(target_vector):
                    population[j] = trial_vector
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial_vector
        return self.f_opt, self.x_opt