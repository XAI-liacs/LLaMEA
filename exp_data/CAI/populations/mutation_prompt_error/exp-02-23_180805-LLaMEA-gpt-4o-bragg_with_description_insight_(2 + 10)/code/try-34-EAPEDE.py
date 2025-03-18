import numpy as np
from scipy.optimize import minimize

class EAPEDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 20
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability
        lb, ub = func.bounds.lb, func.bounds.ub

        population = lb + (ub - lb) * np.random.rand(population_size, self.dim)
        best_idx = np.argmin([func(ind) for ind in population])
        best = population[best_idx].copy()
        eval_count = population_size
        previous_best = func(best)

        while eval_count < self.budget:
            for i in range(population_size):
                if eval_count >= self.budget:
                    break

                indices = np.random.choice(range(population_size), 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + (F + 0.2 * np.random.rand()) * (b - c), lb, ub)

                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial = self.apply_periodicity(trial, lb, ub)

                f_trial = func(trial)
                eval_count += 1

                if f_trial < func(population[i]):
                    population[i] = trial
                    if f_trial < func(best):
                        best = trial
                        CR = max(0.5, CR - 0.05) if f_trial < previous_best else min(1.0, CR + 0.05)
                        previous_best = f_trial

            if eval_count + self.dim <= self.budget:
                bounds = [(lb[i], ub[i]) for i in range(self.dim)]
                res = minimize(lambda x: func(np.clip(x, lb, ub)), best, method='L-BFGS-B', bounds=bounds)
                eval_count += res.nfev
                if res.fun < func(best):
                    best = res.x

        return best

    def apply_periodicity(self, trial, lb, ub):
        period = self.dim // 2
        for i in range(0, self.dim, period):
            period_mean = np.mean(trial[i:i+period])
            trial[i:i+period] = np.clip(period_mean, lb[i:i+period], ub[i:i+period])
        return trial

# Example usage:
# func = YourBlackBoxFunction()
# optimizer = EAPEDE(budget=1000, dim=10)
# best_solution = optimizer(func)