import numpy as np
from scipy.optimize import minimize

class HierarchicalCooperativeOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def island_differential_evolution(self, func, bounds, islands=3, pop_size=10, F=0.5, CR=0.9, max_iter=None):
        if max_iter is None:
            max_iter = self.budget // (islands * pop_size)

        populations = [np.random.uniform(bounds.lb, bounds.ub, (pop_size, self.dim)) for _ in range(islands)]
        scores = [np.empty(pop_size) for _ in range(islands)]

        def periodic_enforcement(x):
            penalty = 0.0
            for i in range(1, len(x)):
                diff = (x[i] - x[i-1]) % (bounds.ub[0] - bounds.lb[0])
                penalty += diff ** 2
            return penalty

        for island in range(islands):
            for i in range(pop_size):
                scores[island][i] = func(populations[island][i]) + periodic_enforcement(populations[island][i])
                self.eval_count += 1

        for _ in range(max_iter):
            for island in range(islands):
                for i in range(pop_size):
                    if self.eval_count >= self.budget:
                        break

                    indices = [idx for idx in range(pop_size) if idx != i]
                    a, b, c = populations[island][np.random.choice(indices, 3, replace=False)]
                    mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                    
                    cross_points = np.random.rand(self.dim) < CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    
                    trial = np.where(cross_points, mutant, populations[island][i])
                    trial_score = func(trial) + periodic_enforcement(trial)
                    self.eval_count += 1

                    if trial_score < scores[island][i]:
                        populations[island][i] = trial
                        scores[island][i] = trial_score

            # Cooperative phase
            best_individuals = [pop[np.argmin(sc)] for pop, sc in zip(populations, scores)]
            cooperative_trial = np.mean(best_individuals, axis=0)
            cooperative_score = func(cooperative_trial) + periodic_enforcement(cooperative_trial)
            self.eval_count += 1

            for island in range(islands):
                if cooperative_score < np.min(scores[island]):
                    worst_idx = np.argmax(scores[island])
                    populations[island][worst_idx] = cooperative_trial
                    scores[island][worst_idx] = cooperative_score

        best_island = np.argmin([np.min(sc) for sc in scores])
        best_idx = np.argmin(scores[best_island])
        return populations[best_island][best_idx]

    def __call__(self, func):
        bounds = func.bounds
        best_solution = self.island_differential_evolution(func, bounds)

        if self.eval_count < self.budget:
            res = minimize(func, best_solution, method='L-BFGS-B',
                           bounds=[(bounds.lb[i], bounds.ub[i]) for i in range(self.dim)],
                           options={'maxfun': self.budget - self.eval_count})
            best_solution = res.x
            self.eval_count += res.nfev

        return best_solution