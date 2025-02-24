import numpy as np
from scipy.optimize import minimize

class PSOHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def _reflectivity_cost(self, params, func):
        if self.eval_count < self.budget:
            self.eval_count += 1
            # Adding a penalty for non-periodic solutions
            penalty = np.sum(np.abs(np.diff(params, n=2)))
            return func(params) + penalty
        else:
            raise RuntimeError("Exceeded budget!")

    def _particle_swarm_optimization(self, func, bounds, num_particles=30, max_iter=200):
        lb, ub = bounds.lb, bounds.ub
        position = np.random.uniform(lb, ub, (num_particles, self.dim))
        velocity = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best_position = position.copy()
        personal_best_value = np.array([self._reflectivity_cost(p, func) for p in position])
        global_best_idx = np.argmin(personal_best_value)
        global_best_position = personal_best_position[global_best_idx]

        for t in range(max_iter):
            inertia_weight = 0.9 - t * (0.5 / max_iter)
            for i in range(num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = 2.0 * r1 * (personal_best_position[i] - position[i])
                social_velocity = 2.0 * r2 * (global_best_position - position[i])
                velocity[i] = inertia_weight * velocity[i] + cognitive_velocity + social_velocity
                position[i] = np.clip(position[i] + velocity[i], lb, ub)
                
                cost = self._reflectivity_cost(position[i], func)
                if cost < personal_best_value[i]:
                    personal_best_value[i] = cost
                    personal_best_position[i] = position[i].copy()

            global_best_idx = np.argmin(personal_best_value)
            global_best_position = personal_best_position[global_best_idx]

        return global_best_position

    def __call__(self, func):
        bounds = func.bounds
        # Global search with Particle Swarm Optimization
        best_global = self._particle_swarm_optimization(func, bounds)

        # Local search using BFGS
        result = minimize(lambda x: self._reflectivity_cost(x, func), best_global, method='L-BFGS-B', bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)])
        
        return result.x if result.success else best_global