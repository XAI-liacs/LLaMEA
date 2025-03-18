import numpy as np
from scipy.optimize import minimize

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

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
                F_adaptive = 0.3 + 0.2 * np.random.rand()  # Adaptive mutation factor range enhanced for diversity
                CR_adaptive = 0.5 + 0.4 * np.random.rand()  # Adaptive crossover rate for improved convergence
                mutant = np.clip(a + F_adaptive * (b - c), lb, ub)
                trial = np.where(np.random.rand(self.dim) < CR_adaptive, mutant, population[j])
                trial_score = func(trial)
                self.eval_count += 1
                if trial_score < scores[j]:
                    population[j] = trial
                    scores[j] = trial_score
                    if trial_score < best_score:
                        best = trial
                        best_score = trial_score

        return best, best_score

    def local_optimization(self, func, x0, bounds):
        if self.eval_count < self.budget * 0.8:  # Selective local search initiation
            res = minimize(func, x0, method='L-BFGS-B', bounds=list(zip(bounds.lb, bounds.ub)))
            return res.x if res.success else x0
        return x0

    def __call__(self, func):
        bounds = func.bounds
        pop_size = 10
        max_iter = self.budget // pop_size
        best_global, _ = self.differential_evolution(func, bounds, pop_size, max_iter=max_iter)
        
        # Local optimization
        best_local = self.local_optimization(func, best_global, bounds)
        if self.eval_count < self.budget:
            return best_local
        else:
            return best_global