import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20  # Adaptive initial population size
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.local_iter = 10
        self.convergence_threshold = 1e-6

    def _differential_evolution(self, func):
        pop_size = self.initial_population_size
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (pop_size, self.dim))
        scores = np.array([self._evaluate(func, ind) for ind in pop])
        evaluations = pop_size
        best_idx = np.argmin(scores)
        best = pop[best_idx, :]

        while evaluations < self.budget:
            for i in range(pop_size):
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), func.bounds.lb, func.bounds.ub)
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i, :])
                trial_score = self._evaluate(func, trial)
                evaluations += 1

                if trial_score < scores[i]:
                    scores[i] = trial_score
                    pop[i, :] = trial

                    if trial_score < scores[best_idx]:
                        best_idx = i
                        best = pop[i, :]

                if evaluations >= self.budget:
                    break

            if np.std(scores) < self.convergence_threshold:
                break

            pop_size = int(self.initial_population_size * (1.0 - evaluations / self.budget))
            pop_size = max(5, pop_size)  # Ensure a minimum population size
            self.mutation_factor = 0.5 + 0.5 * np.std(scores) / (1 + np.std(scores))

        return best

    def _evaluate(self, func, individual):
        evals = [func(individual) for _ in range(5)]
        return np.mean(evals)  # Noise-resistant evaluation

    def _local_optimization(self, func, initial_guess):
        res = minimize(func, initial_guess, method='L-BFGS-B', bounds=list(zip(func.bounds.lb, func.bounds.ub)), options={'maxiter': self.local_iter})
        return res.x

    def __call__(self, func):
        best_solution = self._differential_evolution(func)
        refined_solution = self._local_optimization(func, best_solution)
        return refined_solution