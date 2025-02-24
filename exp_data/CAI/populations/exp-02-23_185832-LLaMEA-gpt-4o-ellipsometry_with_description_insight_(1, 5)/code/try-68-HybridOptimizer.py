import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)

        # Improved chaotic initial sampling for better diversity
        initial_samples = min(self.budget // 2, 10 * self.dim)
        particles = lb + (ub - lb) * np.random.rand(initial_samples, self.dim)
        velocities = np.random.uniform(-0.1, 0.1, (initial_samples, self.dim))
        sample_evals = np.array([func(particle) for particle in particles])

        self.budget -= initial_samples

        global_best_idx = np.argmin(sample_evals)
        global_best_position = particles[global_best_idx]

        if self.budget > 0:
            for i in range(initial_samples):
                r1, r2, r3 = np.random.rand(3)  # Added r3 for adaptive exploration
                velocities[i] += r1 * (particles[i] - global_best_position) + r2 * (global_best_position - particles[i])
                velocities[i] *= (0.9 + r3 * 0.1)  # Adaptive scaling for exploration
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
            perturbation = np.random.normal(0, 0.05, size=self.dim)
            x0_enhanced = np.clip(x0 + perturbation, lb, ub)

            lr = 0.2  # Adjusted initial learning rate
            for _ in range(min(self.budget, 40)):
                grad = np.zeros(self.dim)
                x0_enhanced -= lr * grad
                x0_enhanced = np.clip(x0_enhanced, lb, ub)
                lr *= 0.9  # Adjusted adaptation of the learning rate

            result = minimize(func, x0_enhanced, method='nelder-mead', bounds=list(zip(lb, ub)),
                              options={'maxfev': self.budget, 'xatol': 1e-9, 'fatol': 1e-9})
            return result.x, result.fun

        if self.budget > 0:
            final_sample, final_val = enhanced_local_optimization(global_best_position)
            if final_val < sample_evals[global_best_idx]:
                global_best_position, sample_evals[global_best_idx] = final_sample, final_val

        return global_best_position, sample_evals[global_best_idx]