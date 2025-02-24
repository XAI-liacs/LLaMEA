import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)

        initial_samples = min(self.budget // 2, 10 * self.dim)
        particles = lb + (ub - lb) * np.random.rand(initial_samples, self.dim)
        velocities = np.random.uniform(-0.1, 0.1, (initial_samples, self.dim))
        sample_evals = np.array([func(particle) for particle in particles])

        self.budget -= initial_samples

        global_best_idx = np.argmin(sample_evals)
        global_best_position = particles[global_best_idx]

        if self.budget > 0:
            for i in range(initial_samples):
                r1, r2 = np.random.rand(2)
                velocities[i] = 0.7 * velocities[i] + r1 * (particles[i] - global_best_position) + r2 * (global_best_position - particles[i])
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)
                new_eval = func(particles[i])
                if new_eval < sample_evals[i]:
                    sample_evals[i] = new_eval
                    if new_eval < sample_evals[global_best_idx]:
                        global_best_idx = i
                        global_best_position = particles[i]

        def enhanced_local_optimization(x0):
            nonlocal sample_evals, global_best_idx
            perturbation = np.random.normal(0, 0.03, size=self.dim)
            x0_enhanced = np.clip(x0 + perturbation, lb, ub)

            lr_initial = 0.12
            momentum = np.zeros(self.dim)

            for _ in range(min(self.budget, 40)):
                grad = np.random.normal(0, 0.1, self.dim)
                momentum = 0.85 * momentum + lr_initial * grad
                x0_enhanced -= momentum
                x0_enhanced = np.clip(x0_enhanced, lb, ub)
                lr_initial *= 0.9

            result = minimize(func, x0_enhanced, method='nelder-mead', bounds=list(zip(lb, ub)),
                              options={'maxfev': self.budget, 'xatol': 1e-9, 'fatol': 1e-9})
            return result.x, result.fun

        if self.budget > 0:
            final_sample, final_val = enhanced_local_optimization(global_best_position)
            if final_val < sample_evals[global_best_idx]:
                global_best_position, sample_evals[global_best_idx] = final_sample, final_val

        return global_best_position, sample_evals[global_best_idx]