import numpy as np
import scipy.optimize as opt

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.f = 0.8  # DE Mutation Factor
        self.cr = 0.9  # DE Crossover Rate

    def quasi_oppositional_periodic_init(self, lb, ub):
        """Initialize population with quasi-oppositional strategy and periodicity encouragement."""
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        opp_pop = lb + ub - pop
        for i in range(self.population_size):
            period = self.dim // 2
            pop[i] = np.tile(pop[i][:period], (self.dim // period) + 1)[:self.dim]  # Ensure full coverage
        return np.vstack((pop, opp_pop))

    def differential_evolution(self, pop, func, lb, ub):
        """Perform the DE algorithm to evolve the population."""
        next_pop = np.empty((self.population_size, self.dim))
        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]
            self.f = 0.7 + np.random.rand() * 0.3  # Adaptive Mutation Factor
            mutant = np.clip(a + self.f * (b - c), lb, ub)
            self.cr = 0.85 + np.random.rand() * 0.15  # Refined Adaptive Crossover Rate
            cross_points = np.random.rand(self.dim) < self.cr
            trial = np.where(cross_points, mutant, pop[i])
            next_pop[i] = trial if func(trial) < func(pop[i]) else pop[i]
        return next_pop
    
    def local_search(self, x, func, lb, ub, intensity=0.1):
        """Perform local search using BFGS with adaptive intensity."""
        options = {'maxfun': int(intensity * self.budget // self.population_size)}
        result = opt.minimize(func, x, bounds=opt.Bounds(lb, ub), method='L-BFGS-B', options=options)
        return result.x if result.success else x

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = self.quasi_oppositional_periodic_init(lb, ub)
        pop = pop[np.argsort([func(x) for x in pop])[:self.population_size]]  # Select top half
        
        evaluations = self.population_size * 2
        while evaluations < self.budget:
            self.population_size = max(10, int(self.population_size * 0.9))  # Dynamic Population Adjustment
            pop = self.differential_evolution(pop, func, lb, ub)
            evaluations += self.population_size
            
            intensity = 0.2 + 0.6 * (1 - evaluations / self.budget)  # Adjusted Adaptive intensity range
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                refined = self.local_search(pop[i], func, lb, ub, intensity=intensity)
                if func(refined) < func(pop[i]):
                    pop[i] = refined
                evaluations += 1
        
        best_idx = np.argmin([func(x) for x in pop])
        return pop[best_idx]