import numpy as np
from scipy.optimize import minimize

class CoEvolutionaryBraggOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def quasi_oppositional_de(self, func, bounds, pop_size=20, F=0.5, CR=0.9, max_iter=None):
        if max_iter is None:
            max_iter = self.budget // (2 * pop_size)

        def repair_periodicity(x):
            for i in range(1, len(x)):
                diff = (x[i] - x[i-1]) % (bounds.ub[0] - bounds.lb[0])
                x[i] = x[i-1] + np.round(diff / 0.2) * 0.2
            return np.clip(x, bounds.lb, bounds.ub)

        population = np.random.uniform(bounds.lb, bounds.ub, (pop_size, self.dim))
        opposite_population = bounds.ub + bounds.lb - population
        scores = np.empty(pop_size)

        for i in range(pop_size):
            scores[i] = func(population[i])
            self.eval_count += 1

        for _ in range(max_iter):
            for i in range(pop_size):
                if self.eval_count >= self.budget:
                    break

                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)

                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                trial = repair_periodicity(trial)
                trial_score = func(trial)
                self.eval_count += 1

                if trial_score < scores[i]:
                    population[i] = trial
                    scores[i] = trial_score

            # Opposite population update
            opposite_population = bounds.ub + bounds.lb - population
            for i in range(pop_size):
                opposite_score = func(opposite_population[i])
                self.eval_count += 1
                if opposite_score < scores[i]:
                    population[i] = opposite_population[i]
                    scores[i] = opposite_score

        best_idx = np.argmin(scores)
        return population[best_idx]

    def __call__(self, func):
        bounds = func.bounds
        best_solution = self.quasi_oppositional_de(func, bounds)
        
        if self.eval_count < self.budget:
            res = minimize(func, best_solution, method='L-BFGS-B', bounds=[(bounds.lb[i], bounds.ub[i]) for i in range(self.dim)],
                           options={'maxfun': self.budget - self.eval_count})
            best_solution = res.x
            self.eval_count += res.nfev

        return best_solution