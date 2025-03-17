import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.scale_factor = 0.5
        self.crossover_rate = 0.9
        self.population = None
        self.bounds = None

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def mutate(self, idx, lb, ub):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        # Dynamic scaling factor adjustment
        F = np.random.uniform(0.4, 0.9) 
        mutant = np.clip(a + F * (b - c), lb, ub)
        return mutant

    def crossover(self, target, mutant):
        # Self-adaptive crossover rate
        CR = np.random.rand() if np.random.rand() < 0.1 else self.crossover_rate
        cross_points = np.random.rand(self.dim) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def select(self, target, trial, func):
        target_fitness = func(target)
        trial_fitness = func(trial)
        return trial if trial_fitness < target_fitness else target, min(target_fitness, trial_fitness)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        best_solution = None
        best_fitness = float('inf')

        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                target = self.population[i]
                mutant = self.mutate(i, lb, ub)
                trial = self.crossover(target, mutant)
                selected, fitness = self.select(target, trial, func)
                self.population[i] = selected

                if fitness < best_fitness:
                    best_solution, best_fitness = selected, fitness

                evaluations += 1

        return best_solution