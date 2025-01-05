import numpy as np

class ALDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.best_solution = None
        self.best_value = float('inf')
        self.population = []

    def initialize_population(self, lb, ub):
        return [lb + (ub - lb) * np.random.rand(self.dim) for _ in range(self.population_size)]

    def levy_flight(self, lam=1.5):
        u = np.random.normal(0, 0.1, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / lam))
        return step

    def differential_mutation(self, target, idx, lb, ub, F=0.5, CR=0.9):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + F * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, lb, ub)
        trial = np.copy(target)
        crossover = np.random.rand(self.dim) < CR
        trial[crossover] = mutant[crossover]
        return trial

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.population = self.initialize_population(lb, ub)

        while evaluations < self.budget:
            for i, individual in enumerate(self.population):
                trial = self.differential_mutation(individual, i, lb, ub)
                
                if np.random.rand() < 0.5:
                    trial += self.levy_flight() * (ub - lb) * 0.05
                    trial = np.clip(trial, lb, ub)

                trial_value = func(trial)
                evaluations += 1

                if trial_value < func(individual):
                    self.population[i] = trial

                if trial_value < self.best_value:
                    self.best_value = trial_value
                    self.best_solution = trial.copy()

                if evaluations >= self.budget:
                    break

        return self.best_solution, self.best_value