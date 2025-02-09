import numpy as np
from scipy.optimize import minimize

class AdaptivePSOWithPeriodicConstraints:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 20
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (population_size, self.dim))
        personal_best = population.copy()
        personal_best_f = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_f)]
        global_best_f = np.min(personal_best_f)

        evaluations = population_size
        inertia_weight = 0.7
        cognitive_coeff = 1.5
        social_coeff = 1.5

        while evaluations < self.budget:
            if evaluations + population_size > self.budget:
                break

            for i in range(population_size):
                # Update velocity
                inertia = inertia_weight * velocities[i]
                cognitive = cognitive_coeff * np.random.rand(self.dim) * (personal_best[i] - population[i])
                social = social_coeff * np.random.rand(self.dim) * (global_best - population[i])
                velocities[i] = inertia + cognitive + social

                # Update position
                trial = population[i] + velocities[i]
                trial = np.clip(trial, lb, ub)
                trial = self.apply_periodicity(trial, evaluations)

                # Evaluate trial solution
                f = func(trial)
                evaluations += 1

                # Update personal best
                if f < personal_best_f[i]:
                    personal_best[i] = trial
                    personal_best_f[i] = f

                # Update global best
                if f < global_best_f:
                    global_best = trial
                    global_best_f = f

            # Enhanced Local Search
            noise = np.random.uniform(-0.05, 0.05, self.dim) * (1 / (1 + 0.1 * evaluations))
            opt_result = minimize(func, global_best + noise, method='L-BFGS-B', bounds=list(zip(lb, ub)))
            evaluations += opt_result.nfev
            if opt_result.fun < global_best_f:
                global_best_f = opt_result.fun
                global_best = opt_result.x

        return global_best

    def apply_periodicity(self, solution, evaluation_step):
        period = max(2, self.dim // (1 + (evaluation_step // 100) % 3))
        for i in range(0, self.dim, period):
            solution[i:i + period] = np.mean(solution[i:i + period])
        return solution