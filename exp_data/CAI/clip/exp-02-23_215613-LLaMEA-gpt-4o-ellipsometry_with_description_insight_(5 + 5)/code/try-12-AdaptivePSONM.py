import numpy as np
from scipy.optimize import minimize

class AdaptivePSONM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.particles = 10
        self.inertia = 0.5
        self.cognitive = 1.5
        self.social = 1.5

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        position = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.particles, self.dim))
        velocity = np.random.uniform(-1, 1, (self.particles, self.dim))
        personal_best = np.copy(position)
        personal_best_value = np.array([func(p) for p in position])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        while self.evaluations < self.budget:
            for i in range(self.particles):
                if self.evaluations >= self.budget:
                    break
                velocity[i] = (self.inertia * velocity[i] +
                               self.cognitive * np.random.rand() * (personal_best[i] - position[i]) +
                               self.social * np.random.rand() * (global_best - position[i]))
                position[i] += velocity[i]
                position[i] = np.clip(position[i], bounds[:, 0], bounds[:, 1])
                value = func(position[i])
                self.evaluations += 1
                if value < personal_best_value[i]:
                    personal_best[i] = position[i]
                    personal_best_value[i] = value
                    if value < global_best_value:
                        global_best = position[i]
                        global_best_value = value

            # Increase frequency of Nelder-Mead refinement if budget allows
            if self.evaluations < self.budget and self.evaluations % 2 == 0:  # Change 1: Refinement frequency
                result = minimize(func, global_best, method='Nelder-Mead', options={'maxfev': min(self.budget - self.evaluations, 10)})  # Change 2: Limit maxfev
                if result.fun < global_best_value:
                    global_best = result.x
                    global_best_value = result.fun
                self.evaluations += result.nfev

        return global_best