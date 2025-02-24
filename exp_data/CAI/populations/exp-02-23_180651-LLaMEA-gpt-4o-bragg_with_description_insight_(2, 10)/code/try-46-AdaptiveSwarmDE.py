import numpy as np

class AdaptiveSwarmDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.initial_F = 0.8
        self.initial_CR = 0.9
        self.population = None
        self.best_solution = None
        self.lb = None
        self.ub = None
        self.velocity = None

    def __call__(self, func):
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        self.initialize_population()
        evaluations = 0
        F = self.initial_F
        CR = self.initial_CR

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                target = self.population[i]

                # Mutation and crossover
                trial = self.mutate_and_crossover(i, F, CR)
                trial = self.periodic_resampling(trial)
                trial_fitness = func(trial)
                evaluations += 1

                # Select the better solution
                if trial_fitness > func(target):  # Assume maximization
                    self.population[i] = trial
                    if self.best_solution is None or trial_fitness > func(self.best_solution):
                        self.best_solution = trial

                # Adaptive parameters
                F = self.adaptive_parameter(F)
                CR = self.adaptive_parameter(CR)

        return self.best_solution

    def initialize_population(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))

    def mutate_and_crossover(self, idx, F, CR):
        candidates = list(range(self.population_size))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)

        # Swarm influence via velocity
        mutant = self.population[a] + F * (self.population[b] - self.population[c]) + self.velocity[idx]
        mutant = np.clip(mutant, self.lb, self.ub)

        cross_points = np.random.rand(self.dim) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, self.population[idx])
        return trial

    def periodic_resampling(self, solution):
        period = self.dim // 2  # Fixed period for simplicity
        for start in range(0, self.dim, period):
            pattern = solution[start:start+period]
            for j in range(start, self.dim, period):
                if np.random.rand() < 0.5:  # Probability of enforcing periodicity
                    solution[j:j+period] = pattern
        return solution

    def adaptive_parameter(self, param):
        return np.clip(param + np.random.normal(0, 0.1), 0.5, 1.0)