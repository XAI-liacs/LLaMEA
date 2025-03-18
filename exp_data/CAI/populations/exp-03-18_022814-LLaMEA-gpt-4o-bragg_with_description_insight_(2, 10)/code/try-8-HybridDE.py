import numpy as np
from scipy.optimize import minimize

class HybridDE:
    def __init__(self, budget, dim, population_size=20, mutation_factor=0.8, crossover_prob=0.7):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.evaluations = 0

    def quasi_oppositional_init(self, lb, ub):
        mid = (lb + ub) / 2.0
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        anti_pop = mid + (mid - pop)
        return np.vstack((pop, np.clip(anti_pop, lb, ub)))

    def de_step(self, pop, scores, func, lb, ub):
        for i in range(self.population_size):
            if self.evaluations >= self.budget:
                break
            idxs = np.random.choice(self.population_size*2, 3, replace=False)
            a, b, c = pop[idxs]
            # Adaptive mutation factor
            self.mutation_factor = 0.5 + 0.5 * np.random.random()
            mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
            cross_points = np.random.rand(self.dim) < self.crossover_prob
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, pop[i])
            trial_score = func(trial)
            self.evaluations += 1
            if trial_score < scores[i]:
                scores[i], pop[i] = trial_score, trial

    def local_search(self, best, func, lb, ub):
        res = minimize(func, best, bounds=[(lb[i], ub[i]) for i in range(self.dim)], method='L-BFGS-B')
        return res.x if res.success else best

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = self.quasi_oppositional_init(lb, ub)
        scores = np.array([func(ind) for ind in pop[:self.population_size]])
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            self.de_step(pop, scores, func, lb, ub)
        
        best_idx = np.argmin(scores)
        best = pop[best_idx]
        best = self.local_search(best, func, lb, ub)
        
        return best