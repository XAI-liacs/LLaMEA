import numpy as np
import scipy.optimize as opt

class MultiScaleAdaptiveHybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.f = 0.8  # DE Mutation Factor
        self.cr = 0.9  # DE Crossover Rate
        self.scale_factor = 0.1  # Factor to adjust initial population bounds
        self.periodicity_penalty = 1.0  # Penalty for non-periodic solutions
    
    def adaptive_initialization(self, lb, ub):
        """Initialize population using multi-scale strategy."""
        mid_range = (ub - lb) / 2
        fine_range = mid_range * self.scale_factor
        large_scale_pop = np.random.uniform(lb, ub, (self.population_size // 2, self.dim))
        fine_scale_pop = np.random.uniform(lb + fine_range, ub - fine_range, (self.population_size // 2, self.dim))
        return np.vstack((large_scale_pop, fine_scale_pop))
    
    def differential_evolution(self, pop, func, lb, ub):
        """Perform the DE algorithm with adaptive population size."""
        pop_size = len(pop)
        next_pop = np.empty((pop_size, self.dim))
        for i in range(pop_size):
            indices = list(range(pop_size))
            indices.remove(i)
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]
            adaptive_f = self.f * np.random.rand()  # Adaptive mutation factor
            mutant = np.clip(a + adaptive_f * (b - c), lb, ub)
            adaptive_cr = np.random.rand()  # Adaptive crossover rate
            cross_points = np.random.rand(self.dim) < adaptive_cr
            trial = np.where(cross_points, mutant, pop[i])
            next_pop[i] = trial if self.adjusted_cost(func, trial) < self.adjusted_cost(func, pop[i]) else pop[i]
        return next_pop
    
    def adjusted_cost(self, func, x):
        """Adjust the cost to encourage periodicity."""
        cost = func(x)
        periodic_penalty = self.periodicity_penalty * np.sum(np.abs(np.diff(x)))
        return cost + periodic_penalty
    
    def local_search(self, x, func, lb, ub):
        """Perform local search using BFGS."""
        result = opt.minimize(self.adjusted_cost, x, bounds=opt.Bounds(lb, ub), method='L-BFGS-B')
        return result.x if result.success else x
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = self.adaptive_initialization(lb, ub)
        evaluations = len(pop)
        
        while evaluations < self.budget:
            pop = self.differential_evolution(pop, func, lb, ub)
            evaluations += len(pop)
            
            for i in range(len(pop)):
                if evaluations >= self.budget:
                    break
                refined = self.local_search(pop[i], func, lb, ub)
                if self.adjusted_cost(func, refined) < self.adjusted_cost(func, pop[i]):
                    pop[i] = refined
                evaluations += 1
        
        best_idx = np.argmin([self.adjusted_cost(func, x) for x in pop])
        return pop[best_idx]