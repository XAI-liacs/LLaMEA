import numpy as np

class AdaptivePeriodicExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_base = 0.8
        self.CR_base = 0.9
        self.population = None
        self.best_solution = None
        self.lb = None
        self.ub = None

    def __call__(self, func):
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        self.initialize_population()
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                target = self.population[i]
                trial = self.mutate_and_crossover(i, evaluations / self.budget)
                trial = self.enforce_periodicity(trial, evaluations)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness > func(target):  # Assume maximization
                    self.population[i] = trial
                    if self.best_solution is None or trial_fitness > func(self.best_solution):
                        self.best_solution = trial

        return self.best_solution

    def initialize_population(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))

    def mutate_and_crossover(self, idx, progress):
        F = self.F_base * (1 - progress) + 0.4 * progress  # Adaptive mutation factor
        CR = self.CR_base * (1 - progress) + 0.6 * progress  # Adaptive crossover rate
        
        candidates = list(range(self.population_size))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        chaotic_factor = np.sin(evaluations) # Introduce chaotic sequence
        mutant = self.population[a] + F * chaotic_factor * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, self.lb, self.ub)
        cross_points = np.random.rand(self.dim) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, self.population[idx])
        return trial

    def enforce_periodicity(self, solution, evaluations):
        period = int(2 + (evaluations % 3))  # Dynamically change period size
        for start in range(0, self.dim, period):
            pattern = solution[start:start+period]
            for j in range(start, self.dim, period):
                solution[j:j+period] = pattern
        return solution