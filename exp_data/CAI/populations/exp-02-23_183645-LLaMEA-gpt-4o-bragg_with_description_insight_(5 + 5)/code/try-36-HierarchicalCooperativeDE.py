import numpy as np
from scipy.optimize import minimize

class HierarchicalCooperativeDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.subpop_count = 3
        self.subpop_size = 8
        self.cross_prob = 0.7
        self.diff_weight = 0.8
        self.local_optimization_prob = 0.1

    def _initialize_subpopulations(self, lb, ub):
        subpopulations = []
        for _ in range(self.subpop_count):
            mid_point = (lb + ub) / 2
            half_range = (ub - lb) / 2
            subpop = mid_point + np.random.uniform(-half_range, half_range, (self.subpop_size, self.dim))
            subpopulations.append(subpop)
        return subpopulations

    def _evaluate_subpopulations(self, subpopulations, func):
        return [np.array([func(ind) for ind in subpop]) for subpop in subpopulations]

    def _differential_evolution_step(self, subpop, scores, lb, ub, func):
        for i in range(self.subpop_size):
            indices = [idx for idx in range(self.subpop_size) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = np.clip(subpop[a] + self.diff_weight * (subpop[b] - subpop[c]), lb, ub)
            cross_points = np.random.rand(self.dim) < self.cross_prob
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, subpop[i])
            trial_score = func(trial)
            if trial_score > scores[i]:
                subpop[i] = trial
                scores[i] = trial_score
        return subpop, scores

    def _local_optimization(self, candidate, func, lb, ub):
        def local_func(x):
            return -func(x)

        result = minimize(local_func, candidate, bounds=[(lb[j], ub[j]) for j in range(self.dim)], method='L-BFGS-B', options={'maxiter': 10})
        return result.x if result.success else candidate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        subpopulations = self._initialize_subpopulations(lb, ub)
        scores = self._evaluate_subpopulations(subpopulations, func)

        evaluations = self.subpop_count * self.subpop_size
        while evaluations < self.budget:
            for subpop_idx in range(self.subpop_count):
                subpopulations[subpop_idx], scores[subpop_idx] = self._differential_evolution_step(subpopulations[subpop_idx], scores[subpop_idx], lb, ub, func)
                evaluations += self.subpop_size

                if np.random.rand() < self.local_optimization_prob:
                    idx = np.random.randint(0, self.subpop_size)
                    candidate = subpopulations[subpop_idx][idx]
                    optimized = self._local_optimization(candidate, func, lb, ub)
                    optimized_score = func(optimized)
                    evaluations += 1
                    if optimized_score > scores[subpop_idx][idx]:
                        subpopulations[subpop_idx][idx] = optimized
                        scores[subpop_idx][idx] = optimized_score

            if evaluations < self.budget:
                # Cooperation among subpopulations
                best_individuals = [subpop[np.argmax(score)] for subpop, score in zip(subpopulations, scores)]
                global_best = best_individuals[np.argmax([func(ind) for ind in best_individuals])]
                for subpop_idx in range(self.subpop_count):
                    if np.random.rand() < 0.5:
                        random_idx = np.random.randint(0, self.subpop_size)
                        subpopulations[subpop_idx][random_idx] = global_best
                        scores[subpop_idx][random_idx] = func(global_best)
                        evaluations += 1

        best_subpop = np.argmax([np.max(score) for score in scores])
        best_idx = np.argmax(scores[best_subpop])
        return subpopulations[best_subpop][best_idx]