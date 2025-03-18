import numpy as np
from scipy.optimize import minimize

class HybridBraggOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def hybrid_pso(self, func, bounds, pop_size=20, w=0.5, c1=1.5, c2=1.5, max_iter=None):  
        if max_iter is None:
            max_iter = self.budget // pop_size

        # Particle Swarm Initialization
        population = np.random.uniform(bounds.lb, bounds.ub, (pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (pop_size, self.dim))
        personal_best = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_scores)
        global_best = personal_best[global_best_idx]

        self.eval_count += pop_size

        def periodic_penalty(x):
            penalty = 0.0
            for i in range(1, len(x)):
                diff = abs(x[i] - x[i-1]) % (bounds.ub[0] - bounds.lb[0])
                penalty += (diff - 0.2) ** 2
            return penalty

        for _ in range(max_iter):
            if self.eval_count >= self.budget:
                break
            
            for i in range(pop_size):
                # Update velocity and position
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    w * velocities[i] +
                    c1 * r1 * (personal_best[i] - population[i]) +
                    c2 * r2 * (global_best - population[i])
                )
                population[i] = np.clip(population[i] + velocities[i], bounds.lb, bounds.ub)
                
                # Evaluate and apply penalty
                score = func(population[i]) + periodic_penalty(population[i])
                self.eval_count += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best[i] = population[i]
                    personal_best_scores[i] = score

                    # Update global best
                    if score < personal_best_scores[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[i]

        return global_best

    def __call__(self, func):
        bounds = func.bounds
        best_solution = self.hybrid_pso(func, bounds)
        
        if self.eval_count < self.budget:
            res = minimize(func, best_solution, method='Nelder-Mead', bounds=[(bounds.lb[i], bounds.ub[i]) for i in range(self.dim)],
                           options={'maxfev': self.budget - self.eval_count})
            best_solution = res.x
            self.eval_count += res.nfev

        return best_solution