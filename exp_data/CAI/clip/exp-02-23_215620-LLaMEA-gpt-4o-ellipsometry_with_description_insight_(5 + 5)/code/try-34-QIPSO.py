import numpy as np
from scipy.optimize import minimize

class QIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        num_particles = int(np.log2(self.dim)) * 2
        particles = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_particles, self.dim))
        velocities = np.zeros_like(particles)
        personal_best = particles.copy()
        personal_best_value = np.array([func(p) for p in personal_best])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        while self.evaluations < self.budget:
            for i, particle in enumerate(particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = 0.5 * velocities[i] + r1 * (personal_best[i] - particle) + r2 * (global_best - particle)
                particles[i] = np.clip(particle + velocities[i], bounds[:, 0], bounds[:, 1])
                
                value = func(particles[i])
                self.evaluations += 1
                
                if value < personal_best_value[i]:
                    personal_best[i] = particles[i]
                    personal_best_value[i] = value
                
                if value < global_best_value:
                    global_best = particles[i]
                    global_best_value = value
                
                if self.evaluations >= self.budget:
                    break

            # Periodically apply local search on global best to refine
            if self.evaluations < self.budget:
                result = minimize(func, global_best, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - self.evaluations})
                self.evaluations += result.nfev
                if result.fun < global_best_value:
                    global_best = result.x
                    global_best_value = result.fun

        return global_best