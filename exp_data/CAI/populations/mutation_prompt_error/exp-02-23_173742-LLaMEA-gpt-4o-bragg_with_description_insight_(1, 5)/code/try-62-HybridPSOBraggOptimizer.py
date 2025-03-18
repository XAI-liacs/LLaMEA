import numpy as np
from scipy.optimize import minimize

class HybridPSOBraggOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def dynamic_topology_pso(self, func, bounds, pop_size=30, omega=0.5, phip=1.5, phig=1.5, max_iter=None):
        if max_iter is None:
            max_iter = self.budget // pop_size

        def periodic_penalty(x):
            penalty = 0.0
            for i in range(1, len(x)):
                diff = abs(x[i] - x[i-1]) % (bounds.ub[0] - bounds.lb[0])
                penalty += (diff - 0.2) ** 2
            return penalty
        
        # Initialize particle positions and velocities
        x = np.random.uniform(bounds.lb, bounds.ub, (pop_size, self.dim))
        v = np.random.uniform(-0.1, 0.1, (pop_size, self.dim))
        pbest = x.copy()
        pbest_scores = np.array([func(xi) + periodic_penalty(xi) for xi in x])
        gbest_idx = np.argmin(pbest_scores)
        gbest = pbest[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]
        self.eval_count = pop_size

        for gen in range(max_iter):
            if self.eval_count >= self.budget:
                break
            
            # Dynamic topology adjustment
            neighborhood_size = max(1, int(pop_size * (1 - gen / max_iter)))
            for i in range(pop_size):
                neighbors = np.random.choice(pop_size, neighborhood_size, replace=False)
                local_best_idx = np.argmin(pbest_scores[neighbors])
                local_best = pbest[neighbors[local_best_idx]]

                # Update velocities and positions
                r_p = np.random.rand(self.dim)
                r_g = np.random.rand(self.dim)
                v[i] = omega * v[i] + phip * r_p * (pbest[i] - x[i]) + phig * r_g * (local_best - x[i])
                x[i] = np.clip(x[i] + v[i], bounds.lb, bounds.ub)

                # Evaluate the new positions
                score = func(x[i]) + periodic_penalty(x[i])
                self.eval_count += 1

                # Update personal best
                if score < pbest_scores[i]:
                    pbest[i] = x[i]
                    pbest_scores[i] = score

                    # Update global best
                    if score < gbest_score:
                        gbest = x[i]
                        gbest_score = score

        return gbest

    def __call__(self, func):
        bounds = func.bounds
        best_solution = self.dynamic_topology_pso(func, bounds)

        if self.eval_count < self.budget:
            res = minimize(func, best_solution, method='L-BFGS-B', bounds=[(bounds.lb[i], bounds.ub[i]) for i in range(self.dim)],
                           options={'maxfun': self.budget - self.eval_count})
            best_solution = res.x
            self.eval_count += res.nfev

        return best_solution