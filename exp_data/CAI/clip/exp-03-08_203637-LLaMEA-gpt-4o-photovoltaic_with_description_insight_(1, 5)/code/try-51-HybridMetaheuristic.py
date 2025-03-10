import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def differential_evolution(self, func, bounds, pop_size=20, F=0.8, CR=0.9):
        lb, ub = func.bounds.lb, func.bounds.ub
        norm_pop = np.random.rand(pop_size, self.dim)
        pop = lb + norm_pop * (ub - lb)
        
        best_idx = None
        best_score = float('inf')

        kernel = C(1.0, (1e-3, 1e3)) * RBF(self.dim, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        while self.evaluations < self.budget:
            new_pop = np.empty_like(pop)
            F = 0.5 + 0.3 * (self.budget - self.evaluations) / self.budget
            X_sample, y_sample = pop, np.array([func(ind) for ind in pop])
            gp.fit(X_sample, y_sample)

            for j in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                adaptive_F = F * (1 + np.random.normal(0, 0.1))
                mutant = np.clip(a + adaptive_F * (b - c), lb, ub)
                
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, pop[j])
                
                trial_score = func(trial)
                self.evaluations += 1
                if trial_score < func(pop[j]):
                    new_pop[j] = trial
                else:
                    new_pop[j] = pop[j]
                
                if trial_score < best_score:
                    best_score = trial_score
                    best_idx = j

            pop = new_pop
            CR = 0.9 - 0.4 * (self.evaluations / self.budget)  # Adjust crossover rate based on progress

            if self.evaluations >= self.budget // 2 or \
               best_score - func(pop[best_idx]) < 1e-6:
                break

        return pop[best_idx]

    def local_search(self, func, x0, bounds):
        res = minimize(func, x0, bounds=[(b, ub) for b, ub in zip(bounds[0], bounds[1])], method='L-BFGS-B')
        return res.x

    def __call__(self, func):
        bounds = func.bounds
        best_solution = self.differential_evolution(func, bounds)

        step = max(1, self.dim // 10)
        for new_dim in range(step, self.dim + 1, step):
            sub_bounds = np.array([func.bounds.lb, func.bounds.ub])[:, :new_dim]
            best_solution[:new_dim] = self.local_search(func, best_solution[:new_dim], sub_bounds)

            if self.evaluations >= self.budget:
                break

        return best_solution