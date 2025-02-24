import numpy as np
from scipy.optimize import minimize

class DynamicMultiStrategyOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0
        self.strategy_switch_threshold = 0.1

    def quasi_oppositional_init(self, lb, ub, population_size):
        pop = np.random.uniform(lb, ub, size=(population_size, self.dim))
        opp_pop = lb + ub - pop
        combined_pop = np.vstack((pop, opp_pop))
        return combined_pop

    def differential_evolution(self, func, bounds, pop_size, F=0.5, CR=0.9, max_iter=1000):
        lb, ub = bounds.lb, bounds.ub
        population = self.quasi_oppositional_init(lb, ub, pop_size)
        scores = np.array([func(ind) for ind in population])
        best_idx = np.argmin(scores)
        best = population[best_idx]
        best_score = scores[best_idx]

        for _ in range(max_iter):
            if self.eval_count >= self.budget:
                break
            for j in range(pop_size):
                if self.eval_count >= self.budget:
                    break
                idxs = [idx for idx in range(len(population)) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)
                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[j])
                trial_score = func(trial)
                self.eval_count += 1
                if trial_score < scores[j]:
                    population[j] = trial
                    scores[j] = trial_score
                    if trial_score < best_score:
                        best = trial
                        best_score = trial_score

        return best, best_score

    def particle_swarm_optimization(self, func, bounds, pop_size, w=0.5, c1=1.5, c2=1.5, max_iter=1000):
        lb, ub = bounds.lb, bounds.ub
        positions = np.random.uniform(lb, ub, size=(pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, size=(pop_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(pos) for pos in positions])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        for _ in range(max_iter):
            if self.eval_count >= self.budget:
                break
            r1, r2 = np.random.rand(2)
            velocities = (w * velocities +
                          c1 * r1 * (personal_best_positions - positions) +
                          c2 * r2 * (global_best_position - positions))
            positions = np.clip(positions + velocities, lb, ub)
            scores = np.array([func(pos) for pos in positions])
            self.eval_count += len(positions)
            for i in range(pop_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = positions[i]
                    if scores[i] < global_best_score:
                        global_best_position = positions[i]
                        global_best_score = scores[i]

        return global_best_position, global_best_score

    def local_optimization(self, func, x0, bounds):
        res = minimize(func, x0, method='L-BFGS-B', bounds=list(zip(bounds.lb, bounds.ub)))
        return res.x if res.success else x0

    def __call__(self, func):
        bounds = func.bounds
        pop_size = 10
        max_iter = self.budget // pop_size

        # Start with DE, and dynamically switch to PSO based on performance
        best_global, best_score = self.differential_evolution(func, bounds, pop_size, max_iter=max_iter // 2)

        if best_score > self.strategy_switch_threshold:
            best_global, best_score = self.particle_swarm_optimization(func, bounds, pop_size, max_iter=max_iter // 2)

        # Local optimization
        best_local = self.local_optimization(func, best_global, bounds)
        return best_local if self.eval_count < self.budget else best_global