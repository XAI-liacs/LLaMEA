import numpy as np
from scipy.optimize import minimize

class HybridDEBFGSOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.population = None
        self.bounds = None
    
    def _initialize_population(self, lb, ub):
        self.population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        quasi_oppositional = lb + ub - self.population
        self.population = np.vstack((self.population, quasi_oppositional))
    
    def _evaluate_population(self, func):
        return np.array([func(ind) for ind in self.population])
    
    def _mutate(self, target_idx):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = a + self.mutation_factor * (b - c)
        return np.clip(mutant, self.bounds[0], self.bounds[1])
    
    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial
    
    def _local_search(self, x0, func):
        result = minimize(func, x0, method='L-BFGS-B', bounds=self.bounds)
        return result.x, result.fun
    
    def __call__(self, func):
        self.bounds = np.array([func.bounds.lb, func.bounds.ub])
        self._initialize_population(func.bounds.lb, func.bounds.ub)
        eval_count = 0
        
        while eval_count < self.budget:
            fitness = self._evaluate_population(func)
            eval_count += len(fitness)
            for i in range(self.population_size):
                mutant = self._mutate(i)
                trial = self._crossover(self.population[i], mutant)
                trial_fit = func(trial)
                eval_count += 1
                if trial_fit < fitness[i]:
                    self.population[i] = trial
                    fitness[i] = trial_fit
                if eval_count >= self.budget:
                    break

            # Local optimization on the best candidate
            best_idx = np.argmin(fitness)
            best_candidate = self.population[best_idx]
            refined_candidate, refined_fit = self._local_search(best_candidate, func)
            if refined_fit < fitness[best_idx]:
                self.population[best_idx] = refined_candidate
                fitness[best_idx] = refined_fit
                eval_count += 1
        
        return self.population[np.argmin(fitness)]