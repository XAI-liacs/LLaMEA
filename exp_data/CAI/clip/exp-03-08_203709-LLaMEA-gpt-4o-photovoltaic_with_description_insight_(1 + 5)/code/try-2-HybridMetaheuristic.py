import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size for DE
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = None

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def mutate(self, idx, candidates):
        a, b, c = candidates[np.random.choice(len(candidates), 3, replace=False)]
        mutant = a + self.F * (b - c)
        return np.clip(mutant, bounds.lb, bounds.ub)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        # Enhanced: Use a weighted average of target and mutant for crossover
        trial = (trial + target) / 2
        return trial

    def local_refinement(self, candidate, func, bounds):
        perturbation_strength = 0.05
        for _ in range(5):  # small local search steps
            perturbation = np.random.uniform(-perturbation_strength, perturbation_strength, self.dim)
            trial = np.clip(candidate + perturbation, bounds.lb, bounds.ub)
            if func(trial) > func(candidate):
                candidate = trial
        return candidate

    def __call__(self, func):
        bounds = func.bounds
        self.population = self.initialize_population(bounds)
        fitness = np.array([func(ind) for ind in self.population])
        num_evaluations = self.pop_size

        while num_evaluations < self.budget:
            for i in range(self.pop_size):
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                mutant = self.mutate(i, candidates)
                trial = self.crossover(self.population[i], mutant)
                trial = self.local_refinement(trial, func, bounds)
                
                trial_fitness = func(trial)
                num_evaluations += 1
                if trial_fitness > fitness[i]:
                    self.population[i] = trial
                    fitness[i] = trial_fitness

                if num_evaluations >= self.budget:
                    break

        best_idx = np.argmax(fitness)
        return self.population[best_idx]