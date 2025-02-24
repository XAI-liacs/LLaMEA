import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)

        # Improved chaotic initial sampling using logistic map for better diversity
        def logistic_map(x, r=3.9):
            return r * x * (1 - x)

        chaotic_samples = min(self.budget // 2, 10 * self.dim)
        chaotic_sequence = np.random.rand(chaotic_samples, self.dim)
        for i in range(chaotic_samples):
            chaotic_sequence[i, :] = logistic_map(chaotic_sequence[i, :])

        particles = lb + (ub - lb) * chaotic_sequence
        velocities = np.random.uniform(-0.1, 0.1, (chaotic_samples, self.dim))
        sample_evals = np.array([func(particle) for particle in particles])

        self.budget -= chaotic_samples

        global_best_idx = np.argmin(sample_evals)
        global_best_position = particles[global_best_idx]

        if self.budget > 0:
            for i in range(chaotic_samples):
                r1, r2 = np.random.rand(2)
                velocities[i] = 0.7 * velocities[i] + r1 * (global_best_position - particles[i]) + r2 * (particles[i] - global_best_position)
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)
                new_eval = func(particles[i])
                if new_eval < sample_evals[i]:
                    sample_evals[i] = new_eval
                    if new_eval < sample_evals[global_best_idx]:
                        global_best_idx = i
                        global_best_position = particles[i]

        def dynamic_local_optimization(x0):
            nonlocal sample_evals, global_best_idx
            perturbation = np.random.normal(0, 0.05, size=self.dim)
            x0_perturbed = np.clip(x0 + perturbation, lb, ub)

            lr_initial = 0.1  # Lower initial learning rate
            momentum = np.zeros(self.dim)
            for _ in range(min(self.budget, 50)):
                grad = np.random.normal(0, 0.1, self.dim)  # Stochastic gradient
                momentum = 0.8 * momentum + lr_initial * grad
                x0_perturbed -= momentum
                x0_perturbed = np.clip(x0_perturbed, lb, ub)
                lr_initial *= 0.9  # Adaptive learning rate

            result = minimize(func, x0_perturbed, method='nelder-mead', bounds=list(zip(lb, ub)),
                              options={'maxfev': self.budget, 'xatol': 1e-9, 'fatol': 1e-9})
            return result.x, result.fun

        if self.budget > 0:
            final_sample, final_val = dynamic_local_optimization(global_best_position)
            if final_val < sample_evals[global_best_idx]:
                global_best_position, sample_evals[global_best_idx] = final_sample, final_val

        return global_best_position, sample_evals[global_best_idx]