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
        velocities = np.random.uniform(-0.15, 0.15, (initial_samples, self.dim))  # Adjusted bounds
        sample_evals = np.array([func(particle) for particle in particles])

        self.budget -= initial_samples

        global_best_idx = np.argmin(sample_evals)
        global_best_position = particles[global_best_idx]

        if self.budget > 0:
            for i in range(initial_samples):
                r1, r2 = np.random.rand(2)
                velocities[i] += r1 * (particles[i] - global_best_position) + r2 * (global_best_position - particles[i])
                velocities[i] *= 0.85  # Refined scaling factor
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)
                new_eval = func(particles[i])
                if new_eval < sample_evals[i]:
                    sample_evals[i] = new_eval
                    if new_eval < sample_evals[global_best_idx]:
                        global_best_idx = i
                        global_best_position = particles[i]

        def chaotic_local_optimization(x0):
            nonlocal sample_evals, global_best_idx
            perturbation = np.random.normal(0, 0.05, size=self.dim)
            x0_chaotic = np.clip(x0 + perturbation, lb, ub)

            lr = 0.1
            for _ in range(min(self.budget, 30)):  # Reduced iterations
                grad = np.zeros(self.dim)
                x0_chaotic -= lr * grad
                x0_chaotic = np.clip(x0_chaotic, lb, ub)
                lr *= 0.85  # Further refined learning rate adaptation

            result = minimize(func, x0_chaotic, method='nelder-mead', bounds=list(zip(lb, ub)),
                              options={'maxfev': self.budget, 'xatol': 1e-8, 'fatol': 1e-8})  # Enhanced precision
            return result.x, result.fun

        if self.budget > 0:
            final_sample, final_val = chaotic_local_optimization(global_best_position)
            if final_val < sample_evals[global_best_idx]:
                global_best_position, sample_evals[global_best_idx] = final_sample, final_val

        return global_best_position, sample_evals[global_best_idx]