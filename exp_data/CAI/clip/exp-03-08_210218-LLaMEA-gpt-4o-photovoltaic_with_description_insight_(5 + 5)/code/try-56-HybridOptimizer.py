import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_prob = 0.6  # Adjusted from 0.9 to 0.6
        self.local_iter = 15  # Increased from 10 to 15
        self.convergence_threshold = 1e-5  # Adjusted from 1e-6 to 1e-5

    def _differential_evolution(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in pop])
        evaluations = self.population_size
        best_idx = np.argmin(scores)
        best = pop[best_idx, :]

        while evaluations < self.budget:
            self.population_size = max(5, int(20 * (1 - evaluations / self.budget)))  # Adjust population size
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), func.bounds.lb, func.bounds.ub)
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i, :])
                trial_score = func(trial)
                evaluations += 1

                if trial_score < scores[i]:
                    scores[i] = trial_score
                    pop[i, :] = trial

                    if trial_score < scores[best_idx]:
                        best_idx = i
                        best = pop[i, :]

                if evaluations >= self.budget:
                    break

            # Early stopping if convergence is reached
            if np.std(scores) < self.convergence_threshold:
                break

            self.mutation_factor = 0.5 + 0.5 * np.std(scores) / (1 + np.std(scores))  # Dynamic mutation factor adjustment
            self.crossover_prob = 0.2 + 0.7 * np.exp(-np.std(scores))  # Dynamic crossover probability adjustment

            # Alternate local refinement
            if evaluations % (self.local_iter * self.population_size) < self.population_size:
                refined_solution = self._local_optimization(func, best)
                refined_score = func(refined_solution)
                evaluations += 1
                if refined_score < scores[best_idx]:
                    best = refined_solution

        return best

    def _local_optimization(self, func, initial_guess):
        res = minimize(func, initial_guess, method='L-BFGS-B', bounds=list(zip(func.bounds.lb, func.bounds.ub)), options={'maxiter': self.local_iter})
        return res.x

    def __call__(self, func):
        best_solution = self._differential_evolution(func)
        refined_solution = self._local_optimization(func, best_solution)
        return refined_solution