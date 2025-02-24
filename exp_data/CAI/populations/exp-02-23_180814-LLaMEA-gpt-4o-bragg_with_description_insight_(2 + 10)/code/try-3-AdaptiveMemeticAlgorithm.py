import numpy as np
from scipy.optimize import minimize

class AdaptiveMemeticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.adaptive_rate = 0.05

    def initialize_population(self, lb, ub):
        return lb + np.random.rand(self.population_size, self.dim) * (ub - lb)

    def mutate(self, target_idx, pop):
        candidates = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = np.clip(pop[a] + self.F * (pop[b] - pop[c]), 0, 1)
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        trial = np.where(cross_points, mutant, target)
        return trial

    def local_optimization(self, pop, func, lb, ub):
        for i in range(self.population_size):
            result = minimize(func, pop[i], bounds=np.array(list(zip(lb, ub))), method='L-BFGS-B')
            pop[i] = result.x
        return pop

    def optimize(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = self.initialize_population(lb, ub)
        scores = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(scores)
        global_best = pop[best_idx]
        global_best_score = scores[best_idx]

        evals = self.population_size
        while evals < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(i, pop)
                trial = self.crossover(pop[i], mutant)
                trial_score = func(trial)

                if trial_score < scores[i]:
                    pop[i] = trial
                    scores[i] = trial_score

                    if trial_score < global_best_score:
                        global_best = trial
                        global_best_score = trial_score

            # Apply local optimization periodically
            if evals % (self.population_size * 5) == 0:
                pop = self.local_optimization(pop, func, lb, ub)
                scores = np.array([func(ind) for ind in pop])
                best_idx = np.argmin(scores)
                global_best = pop[best_idx]
                global_best_score = scores[best_idx]

            # Dynamically adapt search parameters
            self.F = max(0.5, self.F - self.adaptive_rate)
            self.CR = min(0.9, self.CR + self.adaptive_rate)

            evals += self.population_size

        return global_best

    def __call__(self, func):
        best_solution = self.optimize(func)
        return best_solution

# Example usage:
# optimizer = AdaptiveMemeticAlgorithm(budget=1000, dim=20)
# best_solution = optimizer(func)