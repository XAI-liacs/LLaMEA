import numpy as np
from cma import CMAEvolutionStrategy

class PSOBraggMirrorOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def pso(self, func, bounds, swarm_size=30, max_iter=None):
        if max_iter is None:
            max_iter = self.budget // swarm_size
        
        def periodic_penalty(x):
            penalty = 0.0
            for i in range(1, len(x)):
                diff = (x[i] - x[i-1]) % (bounds.ub[0] - bounds.lb[0])
                penalty += (diff - 0.2) ** 2
            return penalty

        lb, ub = bounds.lb, bounds.ub
        positions = np.random.uniform(lb, ub, (swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) + periodic_penalty(p) for p in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]

        inertia_weight = 0.5
        cognitive_constant = 1.5
        social_constant = 2.0

        for _ in range(max_iter):
            for i in range(swarm_size):
                if self.eval_count >= self.budget:
                    break

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (inertia_weight * velocities[i]
                                 + cognitive_constant * r1 * (personal_best_positions[i] - positions[i])
                                 + social_constant * r2 * (global_best_position - positions[i]))
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

                score = func(positions[i]) + periodic_penalty(positions[i])
                self.eval_count += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

            global_best_position = personal_best_positions[np.argmin(personal_best_scores)]

        return global_best_position

    def __call__(self, func):
        bounds = func.bounds
        best_solution = self.pso(func, bounds)
        
        if self.eval_count < self.budget:
            cma_options = {'bounds': [bounds.lb, bounds.ub], 'verbose': -9, 'maxfevals': self.budget - self.eval_count}
            es = CMAEvolutionStrategy(best_solution, 0.5, cma_options)
            while not es.stop():
                solutions = es.ask()
                es.tell(solutions, [func(s) for s in solutions])
                self.eval_count += len(solutions)
            best_solution = es.result.xbest

        return best_solution