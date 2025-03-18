import numpy as np
from scipy.optimize import minimize

class HybridPSOBraggMirrorOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def particle_swarm_optimization(self, func, bounds, swarm_size=20, inertia=0.5, cognitive=1.5, social=1.5, max_iter=None):
        if max_iter is None:
            max_iter = self.budget // swarm_size

        def periodic_penalty(x):
            penalty = 0.0
            for i in range(1, len(x)):
                diff = (x[i] - x[i - 1]) % (bounds.ub[0] - bounds.lb[0])
                penalty += diff ** 2
            return penalty

        swarm = np.random.uniform(bounds.lb, bounds.ub, (swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (swarm_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.array([func(x) + periodic_penalty(x) for x in swarm])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]

        for _ in range(max_iter):
            if self.eval_count >= self.budget:
                break

            for i in range(swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    inertia * velocities[i]
                    + cognitive * r1 * (personal_best_positions[i] - swarm[i])
                    + social * r2 * (global_best_position - swarm[i])
                )
                swarm[i] = np.clip(swarm[i] + velocities[i], bounds.lb, bounds.ub)

                score = func(swarm[i]) + periodic_penalty(swarm[i])
                self.eval_count += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = swarm[i]

                if score < personal_best_scores[global_best_idx]:
                    global_best_idx = i
                    global_best_position = swarm[i]

        return global_best_position

    def __call__(self, func):
        bounds = func.bounds
        best_solution = self.particle_swarm_optimization(func, bounds)
        
        if self.eval_count < self.budget:
            res = minimize(func, best_solution, method='Nelder-Mead', bounds=[(bounds.lb[i], bounds.ub[i]) for i in range(self.dim)],
                           options={'maxfev': self.budget - self.eval_count})
            best_solution = res.x
            self.eval_count += res.nfev

        return best_solution