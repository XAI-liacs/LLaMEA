import numpy as np

class ADE_QT:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 20
        self.F = 0.5  # Mutation factor
        self.CR = 0.9  # Crossover probability
        self.population = []

    def initialize_population(self, lb, ub):
        return [lb + (ub - lb) * np.random.rand(self.dim) for _ in range(self.population_size)]

    def mutate(self, target_idx, lb, ub):
        indices = np.arange(self.population_size)
        indices = np.delete(indices, target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
        return np.clip(mutant, lb, ub)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def quantum_tunneling(self, individual, lb, ub):
        tunneling_prob = 0.1
        if np.random.rand() < tunneling_prob:
            delta = np.random.standard_normal(self.dim) * (ub - lb) * 0.1
            individual += delta
            individual = np.clip(individual, lb, ub)
        return individual

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.population = self.initialize_population(lb, ub)
        global_best = None
        global_best_value = float('inf')

        while evaluations < self.budget:
            for i in range(self.population_size):
                target = self.population[i]
                mutant = self.mutate(i, lb, ub)
                trial = self.crossover(target, mutant)
                trial = self.quantum_tunneling(trial, lb, ub)
                
                value = func(trial)
                evaluations += 1
                
                if value < func(target):
                    self.population[i] = trial

                if value < global_best_value:
                    global_best_value = value
                    global_best = trial.copy()

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value