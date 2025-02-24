import numpy as np
from scipy.optimize import minimize

class HybridPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm_size = 20
        max_vel = (ub - lb) * 0.2
        particles = np.random.uniform(lb, ub, (swarm_size, self.dim))
        velocities = np.random.uniform(-max_vel, max_vel, (swarm_size, self.dim))
        personal_best = particles.copy()
        personal_best_fitness = np.array([func(p) for p in particles])
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]

        def update_velocity(vel, part, pers_best, glob_best):
            inertia = 0.5
            cognitive = 2.0 * np.random.rand(self.dim)
            social = 2.0 * np.random.rand(self.dim)
            new_velocity = (inertia * vel +
                            cognitive * (pers_best - part) +
                            social * (glob_best - part))
            return np.clip(new_velocity, -max_vel, max_vel)

        while self.evals < self.budget:
            for i in range(swarm_size):
                velocities[i] = update_velocity(velocities[i], particles[i], personal_best[i], global_best)
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)
                fitness = func(particles[i])
                self.evals += 1

                if fitness < personal_best_fitness[i]:
                    personal_best[i] = particles[i]
                    personal_best_fitness[i] = fitness
                    if fitness < global_best_fitness:
                        global_best = particles[i]
                        global_best_fitness = fitness

                if self.evals >= self.budget:
                    break

        # Local refinement using Nelder-Mead
        def bounded_nelder_mead(local_func, x0, bounds, maxiter):
            res = minimize(
                local_func, x0, method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            x_opt = np.clip(res.x, bounds.lb, bounds.ub)
            return x_opt, res.fun

        if self.evals < self.budget:
            maxiter = self.budget - self.evals
            best_x, best_f = bounded_nelder_mead(func, global_best, func.bounds, maxiter)
        else:
            best_x = global_best

        return best_x