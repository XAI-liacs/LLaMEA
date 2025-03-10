import numpy as np

class LayerAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.7
        self.CR = 0.9
        self.population = np.random.rand(self.pop_size, dim)
        self.best_solution = None
        self.best_score = np.inf
        self.success_history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        evals = 0
        pop = self.population * (ub - lb) + lb

        layers = 10
        while evals < self.budget:
            scores = np.array([func(ind) for ind in pop])
            evals += len(pop)
            
            min_idx = np.argmin(scores)
            if scores[min_idx] < self.best_score:
                self.best_score = scores[min_idx]
                self.best_solution = pop[min_idx].copy()

            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial_score = func(trial)
                evals += 1

                if trial_score < scores[i]:
                    pop[i] = trial
                    scores[i] = trial_score
                    self.success_history.append(trial_score)
                    if trial_score < self.best_score:
                        self.best_score = trial_score
                        self.best_solution = trial.copy()

            self.CR = 0.6 + 0.3 * (evals / self.budget)
            success_rate = len(self.success_history) / (1 + len(self.success_history) + evals) 
            self.F = 0.5 + 0.3 * success_rate  # Dynamic scaling factor based on success rate

            if layers < self.dim:
                layers = min(layers + 2, self.dim)
            
            if evals < self.budget:
                refined_solution = self.local_search(self.best_solution, func, layers, lb, ub)
                evals += layers
                refined_score = func(refined_solution)
                if refined_score < self.best_score:
                    self.best_score = refined_score
                    self.best_solution = refined_solution

        return self.best_solution

    def local_search(self, solution, func, layers, lb, ub):
        perturbed_solution = solution.copy()
        perturbation_range = 0.03 * (1 - self.best_score)
        for _ in range(layers):
            idx = np.random.randint(0, len(solution))
            perturbation = np.random.normal(0, perturbation_range) * (ub[idx] - lb[idx])
            perturbed_solution[idx] = np.clip(perturbed_solution[idx] + perturbation, lb[idx], ub[idx])
        return perturbed_solution