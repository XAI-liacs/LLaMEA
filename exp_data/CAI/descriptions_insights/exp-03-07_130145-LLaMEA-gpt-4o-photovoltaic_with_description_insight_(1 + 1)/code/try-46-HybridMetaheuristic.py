import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def differential_evolution(self, func, bounds, pop_size=20, F=0.8, CR=0.9, max_iter=100):
        pop = np.random.rand(pop_size, self.dim) * (bounds[1] - bounds[0]) + bounds[0]
        fitness = np.array([func(ind) for ind in pop])
        self.eval_count += pop_size

        for _ in range(max_iter):
            if self.eval_count >= self.budget:
                break

            for i in range(pop_size):
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Adaptive Mutation Strategy: Adjust F based on convergence
                adaptive_F = F * (1 - self.eval_count / self.budget)
                mutant = np.clip(a + adaptive_F * (b - c), bounds[0], bounds[1])
                # Adaptive Crossover Strategy: Adjust CR based on convergence
                adaptive_CR = CR * (1 - self.eval_count / self.budget) + 0.1
                cross_points = np.random.rand(self.dim) < adaptive_CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                f_trial = func(trial)
                self.eval_count += 1

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial

        return pop, fitness

    def local_refinement(self, func, x0, bounds):
        result = minimize(func, x0, method='Nelder-Mead', bounds=np.transpose(bounds))
        return result.x, result.fun

    def adaptive_layer_expansion(self, func, bounds):
        num_layers = 10
        best_solution = None
        best_value = np.inf

        while num_layers <= self.dim and self.eval_count < self.budget:
            reduced_dim = min(num_layers, self.dim)
            reduced_bounds = [bounds[0][:reduced_dim], bounds[1][:reduced_dim]]
            pop, fitness = self.differential_evolution(func, reduced_bounds)
            local_best_idx = np.argmin(fitness)
            local_best, local_value = self.local_refinement(func, pop[local_best_idx], reduced_bounds)

            if local_value < best_value:
                best_solution = local_best
                best_value = local_value

            num_layers += 5

        return np.concatenate([best_solution, np.zeros(self.dim - len(best_solution))])

    def __call__(self, func):
        bounds = (np.array(func.bounds.lb), np.array(func.bounds.ub))
        best_solution = self.adaptive_layer_expansion(func, bounds)
        return best_solution