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
    
    def periodicity_induced_search(self, pop, func, lb, ub):
        """Encourage periodicity in solutions."""
        periodic_pop = np.empty((self.population_size, self.dim))
        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]
            # Modify mutation strategy to induce periodicity
            phase_shift = np.random.uniform(-np.pi, np.pi, self.dim)
            mutant = np.clip(a + self.f * (b - c) + np.sin(phase_shift), lb, ub)
            cross_points = np.random.rand(self.dim) < self.cr
            trial = np.where(cross_points, mutant, pop[i])
            periodic_pop[i] = trial if func(trial) < func(pop[i]) else pop[i]
        return periodic_pop
    
    def adaptive_local_search(self, x, func, lb, ub, iteration):
        """Perform adaptive local search with variable precision."""
        epsilon = 1e-4 * (0.9 ** (iteration // 10))  # Adaptive precision
        result = opt.minimize(func, x, bounds=opt.Bounds(lb, ub), method='L-BFGS-B', options={'ftol': epsilon})
        return result.x if result.success else x
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = self.quasi_oppositional_init(lb, ub)
        pop = pop[np.argsort([func(x) for x in pop])[:self.population_size]]  # Select top half
        
        evaluations = self.population_size * 2
        iteration = 0
        while evaluations < self.budget:
            pop = self.periodicity_induced_search(pop, func, lb, ub)
            evaluations += self.population_size
            
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                refined = self.adaptive_local_search(pop[i], func, lb, ub, iteration)
                if func(refined) < func(pop[i]):
                    pop[i] = refined
                evaluations += 1
            
            iteration += 1
        
        best_idx = np.argmin([func(x) for x in pop])
        return pop[best_idx]