import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)

        # Sobol sequence for improved diversity in sampling
        initial_samples = min(self.budget // 2, 10 * self.dim)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        particles = lb + (ub - lb) * sampler.random_base2(m=int(np.log2(initial_samples)))
        sample_evals = np.array([func(particle) for particle in particles])

        self.budget -= initial_samples

        # Update particles towards the best found position
        global_best_idx = np.argmin(sample_evals)
        global_best_position = particles[global_best_idx]

        if self.budget > 0:
            for i in range(initial_samples):
                r1, r2 = np.random.rand(2)
                # Dynamic acceleration adjustment
                velocities = np.random.uniform(-0.05, 0.05, self.dim)
                velocities += r1 * (particles[i] - global_best_position) + r2 * (global_best_position - particles[i])
                velocities *= 0.9  # Adjusted scaling for improved convergence
                particles[i] += velocities
                particles[i] = np.clip(particles[i], lb, ub)
                new_eval = func(particles[i])
                if new_eval < sample_evals[i]:
                    sample_evals[i] = new_eval
                    if new_eval < sample_evals[global_best_idx]:
                        global_best_idx = i
                        global_best_position = particles[i]

        def chaotic_local_optimization(x0):
            nonlocal sample_evals, global_best_idx
            perturbation = np.random.normal(0, 0.02, size=self.dim)  # Adjusted perturbation variance
            x0_chaotic = np.clip(x0 + perturbation, lb, ub)

            # Stochastic gradient descent with chaotic learning rate
            lr = 0.1
            for _ in range(min(self.budget, 50)):
                grad = np.zeros(self.dim)  # Corrected gradient initialization
                x0_chaotic -= lr * grad
                x0_chaotic = np.clip(x0_chaotic, lb, ub)
                lr *= 0.85  # Adjusted chaotic adaptation of learning rate

            result = minimize(func, x0_chaotic, method='nelder-mead', bounds=list(zip(lb, ub)),
                              options={'maxfev': self.budget, 'xatol': 1e-10, 'fatol': 1e-10})  # Improved precision
            return result.x, result.fun

        if self.budget > 0:
            final_sample, final_val = chaotic_local_optimization(global_best_position)
            if final_val < sample_evals[global_best_idx]:
                global_best_position, sample_evals[global_best_idx] = final_sample, final_val

        return global_best_position, sample_evals[global_best_idx]