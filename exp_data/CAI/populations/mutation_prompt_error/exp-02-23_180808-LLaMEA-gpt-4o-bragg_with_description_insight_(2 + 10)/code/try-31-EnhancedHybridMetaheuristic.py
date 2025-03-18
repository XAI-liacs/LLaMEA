import numpy as np
import scipy.optimize as opt

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.f = 0.8  # DE Mutation Factor
        self.cr = 0.9  # DE Crossover Rate
    
    def quasi_oppositional_init(self, lb, ub):
        """Initialize population using quasi-oppositional strategy."""
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        opp_pop = lb + ub - pop
        return np.vstack((pop, opp_pop))
    
    def modify_cost_function(self, func):
        """Modify the cost function to encourage periodicity."""
        def modified_func(x):
            # Penalize deviation from periodicity by calculating the variance across layers
            periodicity_penalty = np.var(x)
            return func(x) + 0.1 * periodicity_penalty  # 0.1 is a weight for penalty
        return modified_func
    
    def differential_evolution(self, pop, func, lb, ub):
        """Perform the DE algorithm to evolve the population."""
        next_pop = np.empty((self.population_size, self.dim))
        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]
            adaptive_f = self.f * np.random.rand()  # Adaptive mutation factor
            mutant = np.clip(a + adaptive_f * (b - c), lb, ub)
            cross_points = np.random.rand(self.dim) < self.cr
            trial = np.where(cross_points, mutant, pop[i])
            next_pop[i] = trial if func(trial) < func(pop[i]) else pop[i]
        return next_pop
    
    def local_search(self, x, func, lb, ub):
        """Perform local search using BFGS from scipy, with adaptive intensification."""
        result = opt.minimize(func, x, bounds=opt.Bounds(lb, ub), method='L-BFGS-B')
        return result.x if result.success else x
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = self.quasi_oppositional_init(lb, ub)
        pop = pop[np.argsort([func(x) for x in pop])[:self.population_size]]  # Select top half
        
        evaluations = self.population_size * 2
        modified_func = self.modify_cost_function(func)
        
        while evaluations < self.budget:
            pop = self.differential_evolution(pop, modified_func, lb, ub)
            evaluations += self.population_size
            
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                # Adaptive intensification for local search
                if np.random.rand() < 0.5:
                    refined = self.local_search(pop[i], modified_func, lb, ub)
                    if func(refined) < func(pop[i]):
                        pop[i] = refined
                evaluations += 1
        
        best_idx = np.argmin([func(x) for x in pop])
        return pop[best_idx]