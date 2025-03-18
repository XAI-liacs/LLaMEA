import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)

        # Initial sampling randomized for better exploration
        initial_samples = min(self.budget // 3, 15 * self.dim)
        particles = lb + (ub - lb) * np.random.rand(initial_samples, self.dim)
        velocities = np.random.uniform(-0.2, 0.2, (initial_samples, self.dim))
        sample_evals = np.array([func(particle) for particle in particles])

        self.budget -= initial_samples

        # Update particles with dynamic velocity adjustment
        global_best_idx = np.argmin(sample_evals)
        global_best_position = particles[global_best_idx]

        if self.budget > 0:
            for i in range(initial_samples):
                r1, r2 = np.random.rand(2)
                velocities[i] = 0.5 * velocities[i] + r1 * (particles[i] - global_best_position) + r2 * (global_best_position - particles[i])
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
            perturbation = np.random.normal(0, 0.01, size=self.dim)
            x0_chaotic = np.clip(x0 + perturbation, lb, ub)

            # Adaptive learning rate for local optimization
            lr = 0.2
            for _ in range(min(self.budget, 60)):
                grad = np.random.normal(0, 0.001, self.dim)
                x0_chaotic -= lr * grad
                x0_chaotic = np.clip(x0_chaotic, lb, ub)
                lr *= 0.92

            result = minimize(func, x0_chaotic, method='nelder-mead', bounds=list(zip(lb, ub)),
                              options={'maxfev': self.budget, 'xatol': 1e-10, 'fatol': 1e-10})
            return result.x, result.fun

        if self.budget > 0:
            final_sample, final_val = chaotic_local_optimization(global_best_position)
            if final_val < sample_evals[global_best_idx]:
                global_best_position, sample_evals[global_best_idx] = final_sample, final_val

        return global_best_position, sample_evals[global_best_idx]